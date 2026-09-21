struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct SpatialVelocity { linear: vec4<f32>, angular: vec4<f32> }
struct Contact { normal_gap: vec4<f32>, material: vec4<f32>, ids: vec4<u32> }
struct Row { inverse_normal: f32, tangent_lipschitz: f32, target_speed: f32, pad: f32, lambda: vec4<f32> }
struct Params {
    dofs: u32, rows: u32, particles: u32, bodies: u32,
    iterations: u32, dt: f32, recovery: f32, max_recovery: f32,
}
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> inverse_mass: array<f32>;
@group(0) @binding(3) var<storage, read> contacts: array<Contact>;
@group(0) @binding(4) var<storage, read> poses: array<Transform>;
@group(0) @binding(5) var<storage, read> body_com: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> jacobians: array<f32>;
@group(0) @binding(7) var<storage, read> mass: array<f32>;
@group(0) @binding(8) var<storage, read_write> rigid_velocity: array<f32>;
@group(0) @binding(9) var<storage, read_write> factor: array<f32>;
@group(0) @binding(10) var<storage, read_write> projected: array<f32>;
@group(0) @binding(11) var<storage, read_write> response: array<f32>;
@group(0) @binding(12) var<storage, read_write> rows: array<Row>;
@group(0) @binding(13) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(14) var<uniform> params: Params;
@group(0) @binding(15) var<storage, read_write> body_velocity: array<SpatialVelocity>;
@group(0) @binding(16) var<storage, read_write> active_rows: array<u32>;
var<workgroup> active_prefix: array<u32, 64>;

fn scan_contacts(row: u32, lane: u32) -> u32 {
    var enabled = 0u;
    if row < params.rows {
        enabled = select(0u, 1u, contacts[row].ids.w == 1u);
    }
    active_prefix[lane] = enabled;
    workgroupBarrier();
    for (var offset = 1u; offset < 64u; offset *= 2u) {
        var previous = 0u;
        if lane >= offset { previous = active_prefix[lane-offset]; }
        workgroupBarrier();
        active_prefix[lane] += previous;
        workgroupBarrier();
    }
    return enabled;
}

@compute @workgroup_size(64)
fn count_contacts(@builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let enabled = scan_contacts(gid.x, lane);
    if lane == 63u { active_rows[params.rows+1u+group.x] = active_prefix[63]; }
}

@compute @workgroup_size(1)
fn offset_contacts() {
    var total = 0u;
    for (var group = 0u; group < (params.rows+63u)/64u; group += 1u) {
        let index = params.rows+1u+group;
        let count = active_rows[index];
        active_rows[index] = total;
        total += count;
    }
    active_rows[0] = total;
}

@compute @workgroup_size(64)
fn scatter_contacts(@builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let enabled = scan_contacts(gid.x, lane);
    if enabled != 0u {
        let index = active_rows[params.rows+1u+group.x] + active_prefix[lane];
        active_rows[index] = gid.x;
    }
}

fn finite(x: f32) -> bool { return abs(x) <= 3.402823e38; }
fn rotate(q: vec4<f32>, p: vec3<f32>) -> vec3<f32> {
    return p + 2.0 * cross(q.xyz, cross(q.xyz, p) + q.w * p);
}
fn basis(normal: vec3<f32>) -> mat3x3<f32> {
    let axis = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.x) < 0.8);
    let tangent = normalize(cross(normal, axis));
    return mat3x3<f32>(normal, tangent, cross(normal, tangent));
}

@compute @workgroup_size(1)
fn factor_mass() {
    if atomicLoad(&status) != 0u { return; }
    let n = params.dofs;
    for (var i = 0u; i < n * n; i += 1u) { factor[i] = 0.0; }
    for (var i = 0u; i < n; i += 1u) {
        for (var j = 0u; j <= i; j += 1u) {
            var value = mass[i*n+j];
            if !finite(value) || abs(value-mass[j*n+i]) > 1e-4 * (1.0+abs(value)) {
                atomicStore(&status, 1u); return;
            }
            for (var k = 0u; k < j; k += 1u) { value -= factor[i*n+k] * factor[j*n+k]; }
            if i == j {
                if mass[i*n+i] == 0.0 {
                    for (var k = 0u; k < n; k += 1u) {
                        if mass[i*n+k] != 0.0 { atomicStore(&status, 1u); return; }
                    }
                } else {
                    if !finite(value) || value <= 0.0 { atomicStore(&status, 1u); return; }
                    factor[i*n+i] = sqrt(value);
                }
            } else if factor[j*n+j] > 0.0 { factor[i*n+j] = value / factor[j*n+j]; }
            else if abs(value) > 1e-10 { atomicStore(&status, 1u); return; }
        }
    }
}

@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= params.rows { return; }
    rows[r] = Row(0.0, 0.0, 0.0, 0.0, vec4<f32>(0.0));
    let contact = contacts[r];
    if contact.ids.w == 0u || atomicLoad(&status) != 0u { return; }
    let particle = contact.ids.x;
    let body = contact.ids.y;
    if contact.ids.w != 1u || particle >= params.particles
        || (body != 0xffffffffu && body >= params.bodies)
        || !finite(contact.normal_gap.w) || !finite(contact.material.x) || contact.material.x < 0.0
        || abs(dot(contact.normal_gap.xyz, contact.normal_gap.xyz) - 1.0) > 1e-4
        || !finite(dot(contact.normal_gap.xyz, contact.normal_gap.xyz)) {
        atomicOr(&status, 2u); return;
    }
    let w = inverse_mass[particle];
    if !finite(w) || w < 0.0 { atomicOr(&status, 2u); return; }
    let n = params.dofs;
    let directions = basis(contact.normal_gap.xyz);
    var offset = vec3<f32>(0.0);
    if body != 0xffffffffu {
        offset = positions[particle].xyz - poses[body].position.xyz - rotate(poses[body].rotation, body_com[body].xyz);
    }
    var diagonal = vec3<f32>(w);
    for (var k = 0u; k < 3u; k += 1u) {
        let base = (r*3u+k)*n;
        var solution: array<f32, 64>;
        for (var d = 0u; d < n; d += 1u) {
            var value = 0.0;
            if body != 0xffffffffu {
                let start = body*6u*n+d;
                let linear = vec3<f32>(jacobians[start], jacobians[start+n], jacobians[start+2u*n]);
                let angular = vec3<f32>(jacobians[start+3u*n], jacobians[start+4u*n], jacobians[start+5u*n]);
                value = dot(directions[k], linear + cross(angular, offset));
            }
            projected[base+d] = value;
            for (var j = 0u; j < d; j += 1u) { value -= factor[d*n+j]*solution[j]; }
            solution[d] = 0.0;
            if factor[d*n+d] > 0.0 { solution[d] = value/factor[d*n+d]; }
            else if abs(value) > 1e-7 { atomicOr(&status, 2u); return; }
        }
        for (var back = 0u; back < n; back += 1u) {
            let d = n - 1u - back;
            var value = solution[d];
            for (var j = d+1u; j < n; j += 1u) { value -= factor[j*n+d]*solution[j]; }
            solution[d] = 0.0;
            if factor[d*n+d] > 0.0 { solution[d] = value/factor[d*n+d]; }
            response[base+d] = solution[d];
            if !finite(solution[d]) { atomicOr(&status, 2u); return; }
        }
        for (var d = 0u; d < n; d += 1u) { diagonal[k] += projected[base+d]*solution[d]; }
    }
    var off_diagonal = 0.0;
    for (var d = 0u; d < n; d += 1u) {
        off_diagonal += projected[(r*3u+2u)*n+d] * response[(r*3u+1u)*n+d];
    }
    let spectral = 0.5 * (diagonal.y+diagonal.z + length(vec2<f32>(diagonal.y-diagonal.z, 2.0*off_diagonal)));
    var target_speed = -contact.normal_gap.w / params.dt;
    if contact.normal_gap.w < 0.0 { target_speed = min(target_speed*params.recovery, params.max_recovery); }
    if !finite(diagonal.x) || diagonal.x < 0.0 || !finite(spectral) || spectral < 0.0 || !finite(target_speed) {
        atomicOr(&status, 2u); return;
    }
    rows[r] = Row(diagonal.x, spectral, target_speed, 0.0, vec4<f32>(0.0));
}

fn relative_velocity(r: u32, direction: vec3<f32>, k: u32) -> f32 {
    var value = dot(velocities[contacts[r].ids.x].xyz, direction);
    for (var d = 0u; d < params.dofs; d += 1u) {
        value -= projected[(r*3u+k)*params.dofs+d] * rigid_velocity[d];
    }
    return value;
}
fn apply_delta(r: u32, direction: vec3<f32>, k: u32, impulse: f32) {
    let particle = contacts[r].ids.x;
    velocities[particle] = vec4<f32>(velocities[particle].xyz + direction * (impulse*inverse_mass[particle]), 0.0);
    for (var d = 0u; d < params.dofs; d += 1u) {
        rigid_velocity[d] -= response[(r*3u+k)*params.dofs+d] * impulse;
    }
}

@compute @workgroup_size(1)
fn solve() {
    if atomicLoad(&status) != 0u { return; }
    for (var iteration = 0u; iteration < params.iterations; iteration += 1u) {
        for (var rank = 0u; rank < active_rows[0]; rank += 1u) {
            let r = active_rows[rank+1u];
            let contact = contacts[r];
            if contact.ids.w != 1u { continue; }
            let directions = basis(contact.normal_gap.xyz);
            let row = rows[r];
            var lambda = row.lambda.xyz;
            if row.inverse_normal > 0.0 {
                lambda.x = max(0.0, lambda.x + (row.target_speed-relative_velocity(r,directions[0],0u))/row.inverse_normal);
            }
            apply_delta(r, directions[0], 0u, lambda.x-row.lambda.x);
            if row.tangent_lipschitz > 0.0 {
                lambda.y -= relative_velocity(r,directions[1],1u)/row.tangent_lipschitz;
                lambda.z -= relative_velocity(r,directions[2],2u)/row.tangent_lipschitz;
            }
            let tangent = length(lambda.yz);
            let limit = contact.material.x*lambda.x;
            if tangent > limit { lambda.y *= limit/tangent; lambda.z *= limit/tangent; }
            apply_delta(r, directions[1], 1u, lambda.y-row.lambda.y);
            apply_delta(r, directions[2], 2u, lambda.z-row.lambda.z);
            rows[r].lambda = vec4<f32>(lambda,0.0);
            if !finite(dot(lambda,lambda)) { atomicStore(&status,4u); return; }
        }
    }
}

@compute @workgroup_size(64)
fn recover_body_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let body = gid.x;
    if body >= params.bodies || atomicLoad(&status) != 0u { return; }
    var result: array<f32, 6>;
    for (var r = 0u; r < 6u; r += 1u) {
        result[r] = 0.0;
        for (var d = 0u; d < params.dofs; d += 1u) {
            result[r] += jacobians[(body*6u+r)*params.dofs+d]*rigid_velocity[d];
        }
    }
    body_velocity[body].linear = vec4<f32>(result[0],result[1],result[2],0.0);
    body_velocity[body].angular = vec4<f32>(result[3],result[4],result[5],0.0);
}
