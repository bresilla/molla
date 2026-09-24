struct Params {
    n: u32, restart: u32, column: u32, mode: u32,
    absolute_tolerance: f32, relative_tolerance: f32, pad0: u32, pad1: u32,
}
struct Diagnostics {
    faults: atomic<u32>, running: u32, used: u32, iterations: u32,
    initial_norm: f32, residual_norm: f32, estimated_norm: f32, inner_threshold: f32,
    rhs_norm: f32, threshold: f32, converged: u32, breakdowns: u32,
}
@group(0) @binding(0) var<storage, read> rhs: array<f32>;
@group(0) @binding(1) var<storage, read> inverse_diagonal: array<f32>;
@group(0) @binding(2) var<storage, read_write> solution: array<f32>;
@group(0) @binding(3) var<storage, read_write> direction: array<f32>;
@group(0) @binding(4) var<storage, read_write> work: array<f32>;
@group(0) @binding(5) var<storage, read_write> basis: array<f32>;
@group(0) @binding(6) var<storage, read_write> small: array<f32>;
@group(0) @binding(7) var<storage, read_write> partials: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> state: Diagnostics;
@group(0) @binding(9) var<uniform> u: Params;
var<workgroup> sums: array<vec4<f32>,64>;

fn finite(value: f32) -> bool { return abs(value) <= 3.402823e38; }
fn h(row: u32, col: u32) -> u32 { return col*(u.restart+1u)+row; }
fn cosines(i: u32) -> u32 { return u.restart*(u.restart+1u)+i; }
fn sines(i: u32) -> u32 { return cosines(u.restart+i); }
fn residuals(i: u32) -> u32 { return sines(u.restart+i); }
fn coefficients(i: u32) -> u32 { return residuals(u.restart+1u+i); }
fn projection(i: u32) -> u32 { return coefficients(u.restart+i); }
fn blocks() -> u32 { return (u.n+63u)/64u; }

fn reduce(lane: u32) {
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride { sums[lane] += sums[lane+stride]; }
        workgroupBarrier();
    }
}

fn scaled_norm(lane: u32, value: vec4<f32>) -> vec4<f32> {
    sums[lane] = abs(value);
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride { sums[lane] = max(sums[lane],sums[lane+stride]); }
        workgroupBarrier();
    }
    let scale = sums[0];
    let normalized = value/select(vec4<f32>(1.0),scale,scale>vec4<f32>(0.0));
    workgroupBarrier();
    sums[lane] = normalized*normalized;
    reduce(lane);
    return scale*sqrt(max(sums[0],vec4<f32>(0.0)));
}

fn total_norm(lane: u32) -> vec4<f32> {
    var scale = vec4<f32>(0.0);
    var squares = vec4<f32>(0.0);
    for (var block = lane; block < blocks(); block += 64u) {
        let value = abs(partials[block]);
        let next = max(scale,value);
        let denominator = select(vec4<f32>(1.0),next,next>vec4<f32>(0.0));
        let a = scale/denominator;
        let b = value/denominator;
        squares = squares*a*a+b*b;
        scale = next;
    }
    return scaled_norm(lane,scale*sqrt(squares));
}

@compute @workgroup_size(64)
fn seed(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < u.n { direction[id.x] = solution[id.x]; }
}

@compute @workgroup_size(64)
fn residual(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>) {
    var terms = vec4<f32>(0.0);
    if id.x < u.n {
        let b = rhs[id.x];
        let raw = b-work[id.x];
        let inverse = inverse_diagonal[id.x];
        let scaled = raw*inverse;
        if !finite(inverse) || inverse == 0.0 { atomicOr(&state.faults,4u); }
        if !finite(b) || !finite(raw) || !finite(scaled) || !finite(solution[id.x]) {
            atomicOr(&state.faults,1u);
        }
        work[id.x] = scaled;
        terms = vec4<f32>(raw,scaled,b,0.0);
    }
    let norms = scaled_norm(lane,terms);
    if lane == 0u { partials[group.x] = norms; }
}

@compute @workgroup_size(64)
fn residual_reduce(@builtin(local_invocation_index) lane: u32) {
    let norms = total_norm(lane).xyz;
    if lane != 0u { return; }
    if !finite(norms.x) || !finite(norms.y) || !finite(norms.z) { atomicOr(&state.faults,1u); }
    state.residual_norm = norms.x;
    state.estimated_norm = norms.y;
    state.rhs_norm = norms.z;
    state.threshold = u.absolute_tolerance+u.relative_tolerance*norms.z;
    state.inner_threshold = max(u.relative_tolerance,4.768372e-7)*norms.y;
    if u.mode == 0u { state.initial_norm = norms.x; }
    state.converged = select(0u,1u,atomicLoad(&state.faults) == 0u && norms.x <= state.threshold);
    state.running = select(0u,1u,atomicLoad(&state.faults) == 0u && state.converged == 0u && norms.y > 0.0);
    state.used = 0u;
    small[residuals(0u)] = norms.y;
    for (var i = 1u; i <= u.restart; i++) { small[residuals(i)] = 0.0; }
}

@compute @workgroup_size(64)
fn initial_basis(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u.n { return; }
    var value = 0.0;
    if state.running != 0u { value = work[id.x]/state.estimated_norm; }
    basis[id.x] = value;
    direction[id.x] = value;
}

@compute @workgroup_size(64)
fn precondition(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u.n { return; }
    var value = 0.0;
    if state.running != 0u {
        value = work[id.x]*inverse_diagonal[id.x];
        if !finite(value) { atomicOr(&state.faults,1u); }
    }
    work[id.x] = value;
}

@compute @workgroup_size(64)
fn dot_partials(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>) {
    var value = 0.0;
    if id.x < u.n && state.running != 0u { value = basis[group.y*u.n+id.x]*work[id.x]; }
    sums[lane] = vec4<f32>(value,0.0,0.0,0.0);
    reduce(lane);
    if lane == 0u { partials[group.y*blocks()+group.x] = sums[0]; }
}

@compute @workgroup_size(64)
fn dot_reduce(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    var sum = 0.0;
    for (var block = lane; block < blocks(); block += 64u) { sum += partials[group.x*blocks()+block].x; }
    sums[lane] = vec4<f32>(sum,0.0,0.0,0.0);
    reduce(lane);
    if lane == 0u {
        let value = sums[0].x;
        if !finite(value) { atomicOr(&state.faults,1u); }
        small[projection(group.x)] = value;
        if u.mode == 0u { small[h(group.x,u.column)] = value; }
        else { small[h(group.x,u.column)] += value; }
    }
}

@compute @workgroup_size(64)
fn subtract(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u.n { return; }
    var value = work[id.x];
    for (var j = 0u; j <= u.column; j++) { value = fma(-small[projection(j)],basis[j*u.n+id.x],value); }
    work[id.x] = value;
}

@compute @workgroup_size(64)
fn norm_partials(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>) {
    var value = 0.0;
    if id.x < u.n { value = work[id.x]; }
    let norms = scaled_norm(lane,vec4<f32>(value,0.0,0.0,0.0));
    if lane == 0u { partials[group.x] = norms; }
}

@compute @workgroup_size(64)
fn arnoldi_reduce(@builtin(local_invocation_index) lane: u32) {
    let beta = total_norm(lane).x;
    if lane != 0u || state.running == 0u { return; }
    let col = u.column;
    if !finite(beta) || atomicLoad(&state.faults) != 0u {
        atomicOr(&state.faults,1u);
        state.running = 0u;
        return;
    }
    small[h(col+1u,col)] = beta;
    for (var i = 0u; i < col; i++) {
        let a = small[h(i,col)];
        let b = small[h(i+1u,col)];
        small[h(i,col)] = small[cosines(i)]*a+small[sines(i)]*b;
        small[h(i+1u,col)] = -small[sines(i)]*a+small[cosines(i)]*b;
    }
    let a = small[h(col,col)];
    let scale = max(abs(a),beta);
    state.iterations++;
    if scale == 0.0 {
        state.running = 0u;
        state.breakdowns++;
        return;
    }
    let norm = scale*length(vec2<f32>(a/scale,beta/scale));
    let cosine = a/norm;
    let sine = beta/norm;
    if !finite(norm) || !finite(cosine) || !finite(sine) {
        atomicOr(&state.faults,1u);
        state.running = 0u;
        return;
    }
    small[cosines(col)] = cosine;
    small[sines(col)] = sine;
    small[h(col,col)] = norm;
    let prior = small[residuals(col)];
    small[residuals(col)] = cosine*prior;
    small[residuals(col+1u)] = -sine*prior;
    state.used = col+1u;
    state.estimated_norm = abs(sine*prior);
    if state.estimated_norm <= state.inner_threshold || beta == 0.0 { state.running = 0u; }
}

@compute @workgroup_size(64)
fn next_basis(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u.n { return; }
    var value = 0.0;
    if state.running != 0u { value = work[id.x]/small[h(u.column+1u,u.column)]; }
    basis[(u.column+1u)*u.n+id.x] = value;
    direction[id.x] = value;
}

@compute @workgroup_size(1)
fn backsolve() {
    if atomicLoad(&state.faults) != 0u { state.used = 0u; return; }
    for (var back = 0u; back < state.used; back++) {
        let i = state.used-1u-back;
        var value = small[residuals(i)];
        for (var j = i+1u; j < state.used; j++) { value = fma(-small[h(i,j)],small[coefficients(j)],value); }
        value /= small[h(i,i)];
        if !finite(value) { atomicOr(&state.faults,1u); state.used = 0u; return; }
        small[coefficients(i)] = value;
    }
}

@compute @workgroup_size(64)
fn update_solution(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u.n { return; }
    var value = solution[id.x];
    for (var j = 0u; j < state.used; j++) { value = fma(small[coefficients(j)],basis[j*u.n+id.x],value); }
    if !finite(value) { atomicOr(&state.faults,1u); return; }
    solution[id.x] = value;
}
