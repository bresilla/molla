struct Wrench { linear: vec4<f32>, angular: vec4<f32> }
struct Island { art_start: u32, art_count: u32, row_start: u32, row_count: u32, dofs: u32, scratch: u32, pad0: u32, pad1: u32 }
struct Art { local: u32, width: u32, mass: u32, pad: u32 }
struct Endpoint { local: u32, width: u32, jacobian: u32, pad: u32 }
struct Projection { parent: Endpoint, child: Endpoint }
struct Row { metadata: vec4<i32>, child_linear: vec4<f32>, child_angular: vec4<f32>,
    parent_linear: vec4<f32>, parent_angular: vec4<f32>, feedback: vec4<f32> }
struct Params { dt: f32, islands: u32, rows: u32, bodies: u32 }
@group(0) @binding(0) var<storage, read> islands: array<Island>;
@group(0) @binding(1) var<storage, read> arts: array<Art>;
@group(0) @binding(2) var<storage, read> projections: array<Projection>;
@group(0) @binding(3) var<storage, read> mass: array<f32>;
@group(0) @binding(4) var<storage, read> jacobians: array<f32>;
@group(0) @binding(5) var<storage, read> rows: array<Row>;
@group(0) @binding(6) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(7) var<storage, read_write> impulses: array<f32>;
@group(0) @binding(8) var<storage, read_write> row_wrenches: array<Wrench>;
@group(0) @binding(9) var<storage, read_write> body_forces: array<Wrench>;
@group(0) @binding(10) var<storage, read> incidence_offsets: array<u32>;
@group(0) @binding(11) var<storage, read> incidence_entries: array<vec2<u32>>;
@group(0) @binding(12) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(13) var<uniform> params: Params;
@group(0) @binding(14) var<storage, read> active_islands: array<u32>;
@group(0) @binding(15) var<storage, read_write> parent_wrenches: array<Wrench>;

fn finite(x: f32) -> bool { return abs(x) <= 3.402823e38; }
fn project(endpoint: Endpoint, column: u32, linear: vec3<f32>, angular: vec3<f32>) -> f32 {
    if column < endpoint.local || column >= endpoint.local+endpoint.width { return 0.0; }
    let offset = endpoint.jacobian+column-endpoint.local;
    let n = endpoint.width;
    return dot(linear,vec3<f32>(jacobians[offset],jacobians[offset+n],jacobians[offset+2u*n]))
        + dot(angular,vec3<f32>(jacobians[offset+3u*n],jacobians[offset+4u*n],jacobians[offset+5u*n]));
}
fn assemble_mass(island: Island, offset: u32, scale: f32) {
    let n = island.dofs;
    for (var i = 0u; i < n*n; i++) { scratch[offset+i] = 0.0; }
    for (var a = 0u; a < island.art_count; a++) {
        let art = arts[island.art_start+a];
        for (var i = 0u; i < art.width; i++) {
            for (var j = 0u; j < art.width; j++) {
                scratch[offset+(art.local+i)*n+art.local+j] = mass[art.mass+i*art.width+j]/scale;
            }
        }
    }
}

@compute @workgroup_size(64)
fn solve_loops(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.islands || atomicLoad(&status) != 0u || active_islands[id.x] == 0u { return; }
    let island = islands[id.x];
    let n = island.dofs;
    let r = island.row_count;
    let factor = island.scratch;
    let eigenvectors = factor+n*n;
    let projected = eigenvectors+n*n;
    let response = projected+r*n;
    let gram = response+r*n;
    let rhs = gram+r*r;
    let lower = rhs+r;
    let upper = lower+r;
    let solution = upper+r;
    assemble_mass(island,factor,1.0);
    var scale = 0.0;
    for (var i = 0u; i < n*n; i++) {
        let value = scratch[factor+i];
        if !finite(value) { atomicOr(&status,32768u); return; }
        scale = max(scale,abs(value));
    }
    var spectral = false;
    var spectral_floor = 0.0;
    if scale > 0.0 {
        for (var i = 0u; i < n*n; i++) { scratch[factor+i] /= scale; }
        for (var i = 0u; i < n; i++) {
            for (var j = 0u; j <= i; j++) {
                var value = scratch[factor+i*n+j];
                for (var k = 0u; k < j; k++) { value -= scratch[factor+i*n+k]*scratch[factor+j*n+k]; }
                if i == j {
                    if value <= 1e-12 || !finite(value) { spectral = true; break; }
                    scratch[factor+i*n+j] = sqrt(value);
                } else { scratch[factor+i*n+j] = value/scratch[factor+j*n+j]; }
            }
            if spectral { break; }
        }
        if !spectral {
            var row_bound = 0.0;
            var column_bound = 0.0;
            for (var i = 0u; i < n; i++) {
                var bound = 1.0;
                for (var j = 0u; j < i; j++) { bound += abs(scratch[factor+i*n+j])*scratch[eigenvectors+j]; }
                scratch[eigenvectors+i] = bound/scratch[factor+i*n+i];
                row_bound = max(row_bound,scratch[eigenvectors+i]);
            }
            for (var k = n; k > 0u; k--) {
                let i = k-1u;
                var bound = 1.0;
                for (var j = i+1u; j < n; j++) { bound += abs(scratch[factor+j*n+i])*scratch[eigenvectors+n+j]; }
                scratch[eigenvectors+n+i] = bound/scratch[factor+i*n+i];
                column_bound = max(column_bound,scratch[eigenvectors+n+i]);
            }
            spectral = !finite(row_bound*column_bound) || row_bound*column_bound*f32(n)*1e-12 >= 1.0;
        }
        if spectral {
            assemble_mass(island,factor,scale);
            for (var i = 0u; i < n; i++) {
                for (var j = 0u; j < n; j++) { scratch[eigenvectors+i*n+j] = select(0.0,1.0,i==j); }
            }
            for (var sweep = 0u; sweep < 48u; sweep++) {
                var changed = false;
                for (var p = 0u; p < n; p++) {
                    for (var q = p+1u; q < n; q++) {
                        let cross = scratch[factor+p*n+q];
                        if abs(cross) <= 1e-12 { continue; }
                        changed = true;
                        let tau = (scratch[factor+q*n+q]-scratch[factor+p*n+p])/(2.0*cross);
                        var tangent = 0.5/tau;
                        if abs(tau) < 1e10 { tangent = select(-1.0,1.0,tau>=0.0)/(abs(tau)+sqrt(1.0+tau*tau)); }
                        let cosine = inverseSqrt(1.0+tangent*tangent);
                        let sine = tangent*cosine;
                        scratch[factor+p*n+p] -= tangent*cross;
                        scratch[factor+q*n+q] += tangent*cross;
                        scratch[factor+p*n+q] = 0.0;
                        scratch[factor+q*n+p] = 0.0;
                        for (var k = 0u; k < n; k++) {
                            if k != p && k != q {
                                let kp = scratch[factor+k*n+p];
                                let kq = scratch[factor+k*n+q];
                                let vp = cosine*kp-sine*kq;
                                let vq = sine*kp+cosine*kq;
                                scratch[factor+k*n+p] = vp; scratch[factor+p*n+k] = vp;
                                scratch[factor+k*n+q] = vq; scratch[factor+q*n+k] = vq;
                            }
                            let kp = scratch[eigenvectors+k*n+p];
                            let kq = scratch[eigenvectors+k*n+q];
                            scratch[eigenvectors+k*n+p] = cosine*kp-sine*kq;
                            scratch[eigenvectors+k*n+q] = sine*kp+cosine*kq;
                        }
                    }
                }
                if !changed { break; }
            }
            var peak = 0.0;
            for (var i = 0u; i < n; i++) { peak = max(peak,scratch[factor+i*n+i]); }
            spectral_floor = peak*1e-12;
            for (var i = 0u; i < n; i++) {
                if scratch[factor+i*n+i] < -peak*1e-5 { atomicOr(&status,32768u); return; }
                for (var j = i+1u; j < n; j++) {
                    if abs(scratch[factor+i*n+j]) > peak*1e-6 { atomicOr(&status,32768u); return; }
                }
            }
        }
    }
    for (var i = 0u; i < r; i++) {
        let row_id = island.row_start+i;
        let row = rows[row_id];
        let endpoints = projections[row_id];
        for (var c = 0u; c < n; c++) {
            var value = 0.0;
            if row.metadata.w != 0 {
                value = project(endpoints.child,c,row.child_linear.xyz,row.child_angular.xyz)
                    + project(endpoints.parent,c,row.parent_linear.xyz,row.parent_angular.xyz);
            }
            scratch[projected+i*n+c] = value;
        }
        for (var c = 0u; c < n; c++) {
            var value = 0.0;
            if scale > 0.0 {
                if spectral {
                    let eigenvalue = scratch[factor+c*n+c];
                    if eigenvalue > spectral_floor {
                        for (var k = 0u; k < n; k++) { value += scratch[eigenvectors+k*n+c]*scratch[projected+i*n+k]; }
                        value /= sqrt(scale)*sqrt(eigenvalue);
                    }
                } else {
                    value = scratch[projected+i*n+c]/sqrt(scale);
                    for (var k = 0u; k < c; k++) { value -= scratch[factor+c*n+k]*scratch[response+i*n+k]; }
                    value /= scratch[factor+c*n+c];
                }
            }
            scratch[response+i*n+c] = value;
        }
    }
    for (var i = 0u; i < r; i++) {
        for (var j = 0u; j < r; j++) {
            var value = 0.0;
            for (var c = 0u; c < n; c++) { value += scratch[response+i*n+c]*scratch[response+j*n+c]; }
            scratch[gram+i*r+j] = value;
        }
    }
    for (var i = 0u; i < r; i++) {
        let row = rows[island.row_start+i];
        let inverse_mass = scratch[gram+i*r+i];
        var effective_mass = 0.0;
        if inverse_mass > 1e-14 { effective_mass = 1.0/inverse_mass; }
        var feedback_scale = 1.0;
        if row.feedback.w != 0.0 { feedback_scale = effective_mass; }
        let stiffness = row.feedback.x*feedback_scale;
        let damping = row.feedback.y*feedback_scale;
        let gain = params.dt*(damping+params.dt*stiffness);
        scratch[rhs+i] = 0.0;
        scratch[lower+i] = 0.0;
        scratch[upper+i] = 0.0;
        scratch[solution+i] = 0.0;
        if row.metadata.w == 2 {
            scratch[gram+i*r+i] += 1.0;
            scratch[lower+i] = row.parent_linear.w*params.dt;
            scratch[upper+i] = scratch[lower+i];
        } else if gain > 0.0 && effective_mass > 0.0 && row.metadata.w != 0 {
            scratch[gram+i*r+i] += 1.0/gain;
            scratch[rhs+i] = -row.child_angular.w
                + (damping*row.feedback.z-stiffness*row.child_linear.w)/(damping+params.dt*stiffness);
            scratch[lower+i] = row.parent_linear.w*params.dt;
            scratch[upper+i] = row.parent_angular.w*params.dt;
        } else { scratch[gram+i*r+i] += 1.0; }
        if !finite(scratch[gram+i*r+i]) || !finite(scratch[rhs+i]) { atomicOr(&status,32768u); return; }
    }
    var converged = false;
    for (var iteration = 0u; iteration < 4096u; iteration++) {
        var settled = true;
        for (var i = 0u; i < r; i++) {
            var residual = scratch[rhs+i];
            for (var j = 0u; j < r; j++) { residual = fma(-scratch[gram+i*r+j],scratch[solution+j],residual); }
            let previous = scratch[solution+i];
            let next = clamp(previous+residual/scratch[gram+i*r+i],scratch[lower+i],scratch[upper+i]);
            scratch[solution+i] = next;
            let update_scale = abs(next)+abs(scratch[rhs+i])/scratch[gram+i*r+i];
            settled = settled && abs(next-previous) <= 4e-7*update_scale+1e-30;
        }
        converged = true;
        for (var i = 0u; i < r; i++) {
            var residual = -scratch[rhs+i];
            for (var j = 0u; j < r; j++) { residual = fma(scratch[gram+i*r+j],scratch[solution+j],residual); }
            let value = scratch[solution+i];
            let tolerance = 2e-7*(1.0+abs(scratch[rhs+i]));
            if scratch[lower+i] == scratch[upper+i] { residual = 0.0; }
            else if value == scratch[lower+i] { residual = min(residual,0.0); }
            else if value == scratch[upper+i] { residual = max(residual,0.0); }
            converged = converged && finite(value) && finite(residual) && abs(residual) <= tolerance;
        }
        converged = converged && settled && iteration >= 3u;
        if converged { break; }
    }
    if !converged { atomicOr(&status,32768u); return; }
    for (var i = 0u; i < r; i++) { impulses[island.row_start+i] = scratch[solution+i]; }
}

@compute @workgroup_size(64)
fn publish_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.rows || atomicLoad(&status) != 0u { return; }
    let row = rows[id.x];
    let force = impulses[id.x]/params.dt;
    let linear = row.child_linear.xyz*force;
    let angular = row.child_angular.xyz*force;
    let parent_linear = row.parent_linear.xyz*force;
    let parent_angular = row.parent_angular.xyz*force;
    if !all(abs(linear) <= vec3<f32>(3.402823e38)) || !all(abs(angular) <= vec3<f32>(3.402823e38)) {
        atomicOr(&status,32768u); return;
    }
    row_wrenches[id.x] = Wrench(vec4<f32>(linear,0.0),vec4<f32>(angular,0.0));
    if !all(abs(parent_linear) <= vec3<f32>(3.402823e38)) || !all(abs(parent_angular) <= vec3<f32>(3.402823e38)) {
        atomicOr(&status,32768u); return;
    }
    parent_wrenches[id.x] = Wrench(vec4<f32>(parent_linear,0.0),vec4<f32>(parent_angular,0.0));
}

@compute @workgroup_size(64)
fn apply_loads(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.bodies || atomicLoad(&status) != 0u { return; }
    var load = body_forces[id.x];
    for (var entry = incidence_offsets[id.x]; entry < incidence_offsets[id.x+1u]; entry++) {
        let record = incidence_entries[entry];
        let row = rows[record.x];
        let force = impulses[record.x]/params.dt;
        if record.y == 0u {
            load.linear += vec4<f32>(row.parent_linear.xyz*force,0.0);
            load.angular += vec4<f32>(row.parent_angular.xyz*force,0.0);
        } else {
            load.linear += vec4<f32>(row.child_linear.xyz*force,0.0);
            load.angular += vec4<f32>(row.child_angular.xyz*force,0.0);
        }
    }
    if !all(abs(load.linear.xyz) <= vec3<f32>(3.402823e38)) || !all(abs(load.angular.xyz) <= vec3<f32>(3.402823e38)) {
        atomicOr(&status,32768u); return;
    }
    body_forces[id.x] = load;
}
