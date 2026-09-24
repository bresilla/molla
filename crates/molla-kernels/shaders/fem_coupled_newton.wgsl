struct Params {
    particles: u32, rows: u32, dofs: u32, dimension: u32,
    alpha: f32, absolute_tolerance: f32, relative_tolerance: f32, armijo: f32,
}
struct Diagnostics {
    faults: u32, running: u32, accepted: u32, iterations: u32,
    initial_norm: f32, residual_norm: f32, base_norm: f32, threshold: f32,
    accepted_alpha: f32, trial_norm: f32, converged: u32, line_evaluations: u32,
    linear_iterations: u32, linear_residual_norm: f32, linear_converged: u32, linear_faults: u32,
}
@group(0) @binding(0) var<storage, read_write> soft_velocity: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> rigid_velocity: array<f32>;
@group(0) @binding(2) var<storage, read_write> impulses: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> soft_direction: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> rigid_direction: array<f32>;
@group(0) @binding(5) var<storage, read_write> impulse_direction: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> material_residual: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> material_action: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> soft_terms: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> rigid_terms: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> contact_terms: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> contact_status: array<u32>;
@group(0) @binding(12) var<storage, read> step: array<f32>;
@group(0) @binding(13) var<storage, read> linear_status: array<u32>;
@group(0) @binding(14) var<storage, read_write> packed_vector: array<f32>;
@group(0) @binding(15) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(16) var<storage, read_write> base: array<f32>;
@group(0) @binding(17) var<storage, read_write> partials: array<vec2<f32>>;
@group(0) @binding(18) var<storage, read_write> state: Diagnostics;
@group(0) @binding(19) var<uniform> u: Params;
@group(0) @binding(20) var<storage, read> inverse_diagonal: array<f32>;
var<workgroup> sums: array<vec2<f32>,64>;

fn finite(value: f32) -> bool { return abs(value) <= 3.402823e38; }
fn soft_end() -> u32 { return 3u*u.particles; }
fn rigid_end() -> u32 { return soft_end()+u.dofs; }
fn scale3(start: u32) -> vec3<f32> { return vec3<f32>(inverse_diagonal[start],inverse_diagonal[start+1u],inverse_diagonal[start+2u]); }
fn packed3(start: u32) -> vec3<f32> { return vec3<f32>(packed_vector[start],packed_vector[start+1u],packed_vector[start+2u])*scale3(start); }
fn trial3(start: u32) -> vec3<f32> {
    return vec3<f32>(base[start],base[start+1u],base[start+2u])
        + u.alpha*vec3<f32>(step[start],step[start+1u],step[start+2u])*scale3(start);
}
fn base3(start: u32) -> vec3<f32> { return vec3<f32>(base[start],base[start+1u],base[start+2u]); }

@compute @workgroup_size(64)
fn zero_directions(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < u.particles { soft_direction[id.x] = vec4<f32>(0.0); }
    if id.x < u.dofs { rigid_direction[id.x] = 0.0; }
    if id.x < u.rows { impulse_direction[id.x] = vec4<f32>(0.0); }
}

@compute @workgroup_size(64)
fn unpack_direction(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < u.particles { soft_direction[id.x] = vec4<f32>(packed3(3u*id.x),0.0); }
    if id.x < u.dofs { rigid_direction[id.x] = packed_vector[soft_end()+id.x]*inverse_diagonal[soft_end()+id.x]; }
    if id.x < u.rows { impulse_direction[id.x] = vec4<f32>(packed3(rigid_end()+3u*id.x),0.0); }
}

fn component(i: u32, action: bool) -> f32 {
    if !finite(inverse_diagonal[i]) || inverse_diagonal[i] == 0.0 { return bitcast<f32>(0x7fc00000u); }
    if contact_status[0] != 0u || contact_status[1] != 0u { return bitcast<f32>(0x7fc00000u); }
    if i < soft_end() {
        let node = i/3u;
        let axis = i%3u;
        if action { return material_action[node][axis]-soft_terms[u.particles+node][axis]; }
        return material_residual[node][axis]-soft_terms[node][axis];
    }
    if i < rigid_end() { return rigid_terms[i-soft_end()][select(0u,1u,action)]; }
    let scalar = i-rigid_end();
    return contact_terms[scalar/3u+select(0u,u.rows,action)][scalar%3u];
}

@compute @workgroup_size(64)
fn pack_rhs(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < u.dimension { rhs[id.x] = -component(id.x,false); }
}

@compute @workgroup_size(64)
fn pack_action(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u.dimension { return; }
    var value = 0.0;
    if state.running != 0u { value = component(id.x,true); }
    packed_vector[id.x] = value;
}

fn norm_reduce(lane: u32, value: f32, bad: f32) -> vec2<f32> {
    sums[lane] = vec2<f32>(abs(value),bad);
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride { sums[lane] = max(sums[lane],sums[lane+stride]); }
        workgroupBarrier();
    }
    let scale = sums[0].x;
    let invalid = sums[0].y;
    let normalized = value/select(1.0,scale,scale>0.0);
    workgroupBarrier();
    sums[lane] = vec2<f32>(normalized*normalized,0.0);
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride { sums[lane].x += sums[lane+stride].x; }
        workgroupBarrier();
    }
    return vec2<f32>(scale*sqrt(max(0.0,sums[0].x)),invalid);
}

@compute @workgroup_size(64)
fn norm_partials(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>) {
    var value = 0.0;
    var bad = 0.0;
    if id.x < u.dimension {
        value = rhs[id.x];
        if !finite(value) { value = 0.0; bad = 1.0; }
    }
    let result = norm_reduce(lane,value,bad);
    if lane == 0u { partials[group.x] = result; }
}

fn total_norm(lane: u32) -> vec2<f32> {
    var scale = 0.0;
    var squares = 0.0;
    var bad = 0.0;
    for (var block = lane; block < (u.dimension+63u)/64u; block += 64u) {
        let value = partials[block];
        let next = max(scale,value.x);
        let denominator = select(1.0,next,next>0.0);
        squares = squares*(scale/denominator)*(scale/denominator)+(value.x/denominator)*(value.x/denominator);
        scale = next;
        bad = max(bad,value.y);
    }
    return norm_reduce(lane,scale*sqrt(squares),bad);
}

@compute @workgroup_size(64)
fn initial_norm(@builtin(local_invocation_index) lane: u32) {
    let result = total_norm(lane);
    if lane != 0u { return; }
    state.initial_norm = result.x;
    state.residual_norm = result.x;
    state.threshold = u.absolute_tolerance+u.relative_tolerance*result.x;
    if result.y != 0.0 || !finite(result.x) { state.faults = 1u; return; }
    state.converged = select(0u,1u,result.x<=state.threshold);
    state.running = 1u-state.converged;
}

@compute @workgroup_size(1)
fn begin_iteration() {
    state.accepted = 0u;
    state.base_norm = state.residual_norm;
}

@compute @workgroup_size(64)
fn capture_base(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= u.dimension { return; }
    if i < soft_end() { base[i] = soft_velocity[i/3u][i%3u]; }
    else if i < rigid_end() { base[i] = rigid_velocity[i-soft_end()]; }
    else { base[i] = impulses[(i-rigid_end())/3u][(i-rigid_end())%3u]; }
    if state.running == 0u { rhs[i] = 0.0; }
}

@compute @workgroup_size(1)
fn check_linear() {
    if state.running == 0u { return; }
    state.linear_iterations += linear_status[3];
    state.linear_residual_norm = bitcast<f32>(linear_status[5]);
    state.linear_converged = linear_status[10];
    state.linear_faults = linear_status[0];
    if linear_status[0] != 0u { state.faults |= 2u; state.running = 0u; }
}

@compute @workgroup_size(64)
fn apply_trial(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < u.particles { soft_direction[id.x] = vec4<f32>(0.0); }
    if id.x < u.dofs { rigid_direction[id.x] = 0.0; }
    if id.x < u.rows { impulse_direction[id.x] = vec4<f32>(0.0); }
    if state.running == 0u || state.accepted != 0u { return; }
    if id.x < u.particles { soft_velocity[id.x] = vec4<f32>(trial3(3u*id.x),soft_velocity[id.x].w); }
    if id.x < u.dofs { rigid_velocity[id.x] = base[soft_end()+id.x]+u.alpha*step[soft_end()+id.x]*inverse_diagonal[soft_end()+id.x]; }
    if id.x < u.rows { impulses[id.x] = vec4<f32>(trial3(rigid_end()+3u*id.x),impulses[id.x].w); }
}

@compute @workgroup_size(64)
fn accept_trial(@builtin(local_invocation_index) lane: u32) {
    let result = total_norm(lane);
    if lane != 0u || state.running == 0u || state.accepted != 0u { return; }
    state.line_evaluations++;
    state.trial_norm = result.x;
    if result.y == 0.0 && finite(result.x) && result.x <= (1.0-u.armijo*u.alpha)*state.base_norm {
        state.accepted = 1u;
        state.iterations++;
        state.accepted_alpha = u.alpha;
        state.residual_norm = result.x;
        state.converged = select(0u,1u,result.x<=state.threshold);
        if state.converged != 0u { state.running = 0u; }
    }
}

@compute @workgroup_size(1)
fn finish_iteration() {
    if state.running != 0u && state.accepted == 0u { state.faults |= 4u; state.running = 0u; }
}

@compute @workgroup_size(64)
fn restore_rejected(@builtin(global_invocation_id) id: vec3<u32>) {
    if state.accepted != 0u { return; }
    if id.x < u.particles { soft_velocity[id.x] = vec4<f32>(base3(3u*id.x),soft_velocity[id.x].w); }
    if id.x < u.dofs { rigid_velocity[id.x] = base[soft_end()+id.x]; }
    if id.x < u.rows { impulses[id.x] = vec4<f32>(base3(rigid_end()+3u*id.x),impulses[id.x].w); }
}

@compute @workgroup_size(64)
fn finish_solve(@builtin(local_invocation_index) lane: u32) {
    let result = total_norm(lane);
    if lane != 0u { return; }
    state.running = 0u;
    state.residual_norm = result.x;
    if result.y != 0.0 || !finite(result.x) { state.faults |= 1u; }
    state.converged = select(0u,1u,state.faults == 0u && result.x<=state.threshold);
}
