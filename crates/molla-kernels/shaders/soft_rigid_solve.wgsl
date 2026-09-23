struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct SpatialVelocity { linear: vec4<f32>, angular: vec4<f32> }
struct Contact {
    normal_gap: vec4<f32>, material: vec4<f32>, ids: vec4<u32>,
    support: vec4<u32>, weights: vec4<f32>,
}
struct Row { inverse_normal: f32, tangent_lipschitz: f32, target_speed: f32, pad: f32, lambda: vec4<f32> }
struct Params {
    dofs: u32, rows: u32, particles: u32, bodies: u32,
    iterations: u32, dt: f32, recovery: f32, max_recovery: f32,
    base_row: u32, pad0: u32, pad1: u32, pad2: u32,
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
        enabled = select(0u, 1u, contacts[row].ids.w == 1u || contacts[row].ids.w == 3u);
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
fn support_node(contact: Contact, i: u32) -> u32 {
    if i == 0u { return contact.ids.x; }
    return contact.support[i-1u];
}
fn rotate(q: vec4<f32>, p: vec3<f32>) -> vec3<f32> {
    return p + 2.0 * cross(q.xyz, cross(q.xyz, p) + q.w * p);
}
fn basis(normal: vec3<f32>) -> mat3x3<f32> {
    let axis = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.x) < 0.8);
    let tangent = normalize(cross(normal, axis));
    return mat3x3<f32>(normal, tangent, cross(normal, tangent));
}

fn anchor(body: u32, local: vec3<f32>) -> vec3<f32> {
    return poses[body].position.xyz + rotate(poses[body].rotation, local);
}
fn valid_body(body: u32) -> bool {
    let norm = dot(poses[body].rotation, poses[body].rotation);
    return finite(dot(poses[body].position.xyz, poses[body].position.xyz))
        && finite(dot(body_com[body].xyz, body_com[body].xyz)) && finite(norm) && abs(norm-1.0) <= 1e-4;
}
fn point_jacobian(body: u32, offset: vec3<f32>, d: u32) -> vec3<f32> {
    let n = params.dofs;
    let start = body*6u*n+d;
    let linear = vec3<f32>(jacobians[start], jacobians[start+n], jacobians[start+2u*n]);
    let angular = vec3<f32>(jacobians[start+3u*n], jacobians[start+4u*n], jacobians[start+5u*n]);
    return linear + cross(angular, offset);
}
fn loop_target(contact: Contact) -> vec3<f32> {
    let error = anchor(contact.ids.x, contact.material.xyz) - anchor(contact.ids.y, contact.weights.xyz);
    let speed = -error * (params.recovery / params.dt);
    return clamp(speed, vec3<f32>(-params.max_recovery), vec3<f32>(params.max_recovery));
}

@compute @workgroup_size(1)
fn factor_mass() {
    if atomicLoad(&status) != 0u { return; }
    let n = params.dofs;
    for (var i = 0u; i < n; i += 1u) {
        if !finite(rigid_velocity[i]) { atomicOr(&status, 4u); return; }
    }
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
    let r = params.base_row + gid.x;
    if r >= params.rows { return; }
    rows[r] = Row(0.0, 0.0, 0.0, 0.0, vec4<f32>(0.0));
    let contact = contacts[r];
    if contact.ids.w == 0u || atomicLoad(&status) != 0u { return; }
    if contact.ids.w == 4u {atomicOr(&status,16u); return;}
    let body = contact.ids.y;
    let ball = contact.ids.w == 3u;
    if ball {
        if contact.ids.x >= params.bodies || body >= params.bodies || contact.ids.x == body
            || contact.support.z != 0u || !finite(dot(contact.material.xyz, contact.material.xyz))
            || !finite(dot(contact.weights.xyz, contact.weights.xyz)) {
            atomicOr(&status, 2u); return;
        }
        if !valid_body(contact.ids.x) || !valid_body(body) {
            atomicOr(&status, 2u); return;
        }
    } else if contact.ids.w != 1u || contact.support.z == 0u || contact.support.z > 3u
        || (body != 0xffffffffu && body >= params.bodies)
        || !finite(contact.normal_gap.w) || !finite(contact.material.x) || contact.material.x < 0.0
        || abs(dot(contact.normal_gap.xyz, contact.normal_gap.xyz) - 1.0) > 1e-4
        || !finite(dot(contact.normal_gap.xyz, contact.normal_gap.xyz)) {
        atomicOr(&status, 2u); return;
    }
    var w = 0.0;
    var point = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var i = 0u; i < contact.support.z; i += 1u) {
        let particle = support_node(contact, i);
        let weight = contact.weights[i];
        if particle >= params.particles || !finite(weight) || weight < 0.0 || weight > 1.0 {
            atomicOr(&status, 2u); return;
        }
        for (var j = 0u; j < i; j += 1u) {
            if particle == support_node(contact, j) { atomicOr(&status, 2u); return; }
        }
        let inverse = inverse_mass[particle];
        if !finite(inverse) || inverse < 0.0 { atomicOr(&status, 2u); return; }
        w += weight * weight * inverse;
        point += weight * positions[particle].xyz;
        weight_sum += weight;
    }
    if (!ball && abs(weight_sum-1.0) > 1e-5) || !finite(dot(point, point)) {
        atomicOr(&status, 2u); return;
    }
    let n = params.dofs;
    if body == 0xffffffffu {
        for (var d = 0u; d < 3u*n; d += 1u) {
            projected[r*3u*n+d] = 0.0;
            response[r*3u*n+d] = 0.0;
        }
        var target_speed = -contact.normal_gap.w / params.dt;
        if contact.normal_gap.w < 0.0 { target_speed = min(target_speed*params.recovery, params.max_recovery); }
        if !finite(w) || w < 0.0 || !finite(target_speed) { atomicOr(&status, 2u); return; }
        rows[r] = Row(w, w, target_speed, 0.0, vec4<f32>(0.0));
        return;
    }
    var directions = basis(contact.normal_gap.xyz);
    var offset_a = vec3<f32>(0.0);
    if ball {
        directions = mat3x3<f32>(vec3<f32>(1.0,0.0,0.0), vec3<f32>(0.0,1.0,0.0), vec3<f32>(0.0,0.0,1.0));
        point = anchor(body, contact.weights.xyz);
        offset_a = rotate(poses[contact.ids.x].rotation, contact.material.xyz - body_com[contact.ids.x].xyz);
    }
    var offset = vec3<f32>(0.0);
    if body != 0xffffffffu {
        offset = point - poses[body].position.xyz - rotate(poses[body].rotation, body_com[body].xyz);
    }
    var diagonal = vec3<f32>(w);
    for (var k = 0u; k < 3u; k += 1u) {
        let base = (r*3u+k)*n;
        var solution: array<f32, 64>;
        for (var d = 0u; d < n; d += 1u) {
            var value = 0.0;
            if body != 0xffffffffu {
                value = dot(directions[k], point_jacobian(body, offset, d));
            }
            if ball { value -= dot(directions[k], point_jacobian(contact.ids.x, offset_a, d)); }
            if !finite(value) { atomicOr(&status, 2u); return; }
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
    if ball {
        let desired = loop_target(contact);
        if !finite(dot(diagonal,diagonal)) || any(diagonal < vec3<f32>(0.0)) || !finite(dot(desired,desired)) {
            atomicOr(&status, 2u); return;
        }
        rows[r] = Row(diagonal.x, diagonal.y, diagonal.z, 0.0, vec4<f32>(0.0));
        return;
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
    let contact = contacts[r];
    var value = 0.0;
    for (var i = 0u; i < contact.support.z; i += 1u) {
        value += contact.weights[i] * dot(velocities[support_node(contact, i)].xyz, direction);
    }
    if contact.ids.y != 0xffffffffu {
        for (var d = 0u; d < params.dofs; d += 1u) {
            value -= projected[(r*3u+k)*params.dofs+d] * rigid_velocity[d];
        }
    }
    return value;
}
fn apply_delta(r: u32, direction: vec3<f32>, k: u32, impulse: f32) {
    let contact = contacts[r];
    for (var i = 0u; i < contact.support.z; i += 1u) {
        let particle = support_node(contact, i);
        velocities[particle] = vec4<f32>(velocities[particle].xyz + direction * (impulse*contact.weights[i]*inverse_mass[particle]), 0.0);
    }
    if contact.ids.y != 0xffffffffu {
        for (var d = 0u; d < params.dofs; d += 1u) {
            rigid_velocity[d] -= response[(r*3u+k)*params.dofs+d] * impulse;
        }
    }
}

@compute @workgroup_size(1)
fn solve() {
    if atomicLoad(&status) != 0u { return; }
    for (var iteration = 0u; iteration < params.iterations; iteration += 1u) {
        for (var rank = 0u; rank < active_rows[0]; rank += 1u) {
            let r = active_rows[rank+1u];
            let contact = contacts[r];
            if contact.ids.w == 3u {
                let row = rows[r];
                let diagonal = vec3<f32>(row.inverse_normal, row.tangent_lipschitz, row.target_speed);
                let desired = loop_target(contact);
                var lambda = row.lambda.xyz;
                for (var k = 0u; k < 3u; k += 1u) {
                    let residual = desired[k] - relative_velocity(r,vec3<f32>(0.0),k);
                    if !finite(residual) { atomicStore(&status,4u); return; }
                    if diagonal[k] == 0.0 {
                        if residual != 0.0 { atomicStore(&status,2u); return; }
                        continue;
                    }
                    let impulse = residual / diagonal[k];
                    if !finite(impulse) { atomicStore(&status,4u); return; }
                    apply_delta(r, vec3<f32>(0.0), k, impulse);
                    lambda[k] += impulse;
                }
                if !finite(dot(lambda,lambda)) { atomicStore(&status,4u); return; }
                rows[r].lambda = vec4<f32>(lambda,0.0);
                continue;
            }
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
fn check_residuals(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= params.rows { return; }
    let contact = contacts[r];
    if contact.ids.w != 1u && contact.ids.w != 3u { return; }
    let row = rows[r];
    var error = 0.0;
    if contact.ids.w == 3u {
        let desired = loop_target(contact);
        for (var k = 0u; k < 3u; k += 1u) {
            let value = abs(desired[k] - relative_velocity(r, vec3<f32>(0.0), k));
            if !finite(value) { atomicOr(&status, 4u); return; }
            error = max(error, value);
        }
    } else {
        let directions = basis(contact.normal_gap.xyz);
        let normal_error = row.target_speed - relative_velocity(r, directions[0], 0u);
        let tangent_velocity = vec2<f32>(relative_velocity(r, directions[1], 1u), relative_velocity(r, directions[2], 2u));
        if !finite(normal_error) || !all(abs(tangent_velocity) <= vec2<f32>(3.402823e38)) {
            atomicOr(&status, 4u); return;
        }
        error = max(0.0, normal_error);
        if row.inverse_normal > 0.0 {
            error = abs(max(0.0, row.lambda.x + normal_error / row.inverse_normal) - row.lambda.x) * row.inverse_normal;
        }
        if row.tangent_lipschitz > 0.0 {
            var trial = row.lambda.yz - tangent_velocity / row.tangent_lipschitz;
            let limit = contact.material.x * row.lambda.x;
            let magnitude = length(trial);
            if !finite(magnitude) || !finite(limit) { atomicOr(&status, 4u); return; }
            if magnitude > limit { trial *= limit / magnitude; }
            error = max(error, length(trial - row.lambda.yz) * row.tangent_lipschitz);
        }
    }
    if !finite(error) { atomicOr(&status, 4u); return; }
    rows[r].pad = error;
    if error > bitcast<f32>(params.pad0) { atomicOr(&status, 64u); }
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
