struct Contact {
    normal_gap: vec4<f32>, material: vec4<f32>, ids: vec4<u32>,
    support: vec4<u32>, weights: vec4<f32>,
}
struct Row { normal: f32, tangent: f32, target_speed: f32, error: f32, lambda: vec4<f32> }
struct Pose { position: vec4<f32>, rotation: vec4<f32> }
struct Uniforms {
    particles: u32, rows: u32, dofs: u32, bodies: u32,
    dt: f32, recovery: f32, max_recovery: f32, pad: u32,
}
@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read> rows: array<Row>;
@group(0) @binding(2) var<storage, read> projected: array<f32>;
@group(0) @binding(3) var<storage, read> response: array<f32>;
@group(0) @binding(4) var<storage, read> poses: array<Pose>;
@group(0) @binding(5) var<storage, read> inverse_mass: array<f32>;
@group(0) @binding(6) var<storage, read> soft_velocity: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> rigid_velocity: array<f32>;
@group(0) @binding(8) var<storage, read> rigid_reference: array<f32>;
@group(0) @binding(9) var<storage, read> soft_direction: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> rigid_direction: array<f32>;
@group(0) @binding(11) var<storage, read> impulses: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> impulse_direction: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> adjacency: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> partials: array<vec2<f32>>;
@group(0) @binding(15) var<storage, read_write> soft_terms: array<vec4<f32>>;
@group(0) @binding(16) var<storage, read_write> rigid_terms: array<vec4<f32>>;
@group(0) @binding(17) var<storage, read_write> contact_terms: array<vec4<f32>>;
@group(0) @binding(18) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(19) var<uniform> u: Uniforms;
var<workgroup> reductions: array<vec2<f32>,64>;

fn finite(x: f32) -> bool { return abs(x) <= 3.402823e38; }
fn finite3(x: vec3<f32>) -> bool { return all(abs(x) <= vec3<f32>(3.402823e38)); }
fn enabled(kind: u32) -> bool { return kind == 1u || kind == 3u || kind == 5u; }
fn node_at(c: Contact, i: u32) -> u32 {
    if i == 0u { return c.ids.x; }
    return c.support[i-1u];
}
fn basis(normal: vec3<f32>) -> mat3x3<f32> {
    let axis = select(vec3<f32>(0.0,1.0,0.0),vec3<f32>(1.0,0.0,0.0),abs(normal.x)<0.8);
    let tangent = normalize(cross(normal,axis));
    return mat3x3<f32>(normal,tangent,cross(normal,tangent));
}
fn rotate(q: vec4<f32>, p: vec3<f32>) -> vec3<f32> {
    return p+2.0*cross(q.xyz,cross(q.xyz,p)+q.w*p);
}
fn loop_target(c: Contact) -> vec3<f32> {
    let a = poses[c.ids.x].position.xyz+rotate(poses[c.ids.x].rotation,c.material.xyz);
    let b = poses[c.ids.y].position.xyz+rotate(poses[c.ids.y].rotation,c.weights.xyz);
    return clamp((b-a)*(u.recovery/u.dt),vec3<f32>(-u.max_recovery),vec3<f32>(u.max_recovery));
}
fn offsets(node: u32) -> u32 { return u.particles+node; }
fn entries(slot: u32) -> u32 { return 2u*u.particles+1u+slot; }

@compute @workgroup_size(64)
fn count_supports(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= u.rows { return; }
    let c = contacts[r];
    if !enabled(c.ids.w) {
        if c.ids.w != 0u { atomicOr(&status[0],2u); }
        return;
    }
    if !finite3(rows[r].lambda.xyz) { atomicOr(&status[0],4u); return; }
    if c.ids.w == 3u {
        if c.support.z != 0u || c.ids.x >= u.bodies || c.ids.y >= u.bodies {
            atomicOr(&status[0],2u);
        }
        return;
    }
    if c.ids.w == 5u {
        if c.support.z != 0u || !(rows[r].normal > 0.0) { atomicOr(&status[0],2u); }
        return;
    }
    if c.support.z == 0u || c.support.z > 3u || !(rows[r].normal > 0.0)
        || !(rows[r].tangent > 0.0) || !finite(rows[r].normal) || !finite(rows[r].tangent)
        || !finite(c.material.x) || c.material.x < 0.0 {
        atomicOr(&status[0],2u); return;
    }
    var weight_sum = 0.0;
    for (var i = 0u; i < c.support.z; i++) {
        let node = node_at(c,i);
        if node >= u.particles || !finite(c.weights[i]) || c.weights[i] < 0.0 || c.weights[i] > 1.0 {
            atomicOr(&status[0],2u); return;
        }
        for (var j = 0u; j < i; j++) {
            if node == node_at(c,j) { atomicOr(&status[0],2u); return; }
        }
        weight_sum += c.weights[i];
        atomicAdd(&adjacency[node],1u);
    }
    if abs(weight_sum-1.0) > 1e-5 { atomicOr(&status[0],2u); }
}

@compute @workgroup_size(1)
fn prefix_supports() {
    var total = 0u;
    for (var node = 0u; node < u.particles; node++) {
        atomicStore(&adjacency[offsets(node)],total);
        total += atomicLoad(&adjacency[node]);
        atomicStore(&adjacency[node],0u);
    }
    atomicStore(&adjacency[offsets(u.particles)],total);
    if total > 3u*u.rows { atomicOr(&status[0],2u); }
}

@compute @workgroup_size(64)
fn scatter_supports(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= u.rows || atomicLoad(&status[0]) != 0u { return; }
    let c = contacts[r];
    if c.ids.w != 1u { return; }
    for (var i = 0u; i < c.support.z; i++) {
        let node = node_at(c,i);
        let slot = atomicLoad(&adjacency[offsets(node)])+atomicAdd(&adjacency[node],1u);
        atomicStore(&adjacency[entries(slot)],3u*r+i);
    }
}

@compute @workgroup_size(64)
fn sort_supports(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node = gid.x;
    if node >= u.particles || atomicLoad(&status[0]) != 0u { return; }
    if !finite(inverse_mass[node]) || inverse_mass[node] < 0.0 { atomicOr(&status[0],2u); return; }
    let start = atomicLoad(&adjacency[offsets(node)]);
    let end = atomicLoad(&adjacency[offsets(node+1u)]);
    for (var i = start+1u; i < end; i++) {
        let value = atomicLoad(&adjacency[entries(i)]);
        var j = i;
        while j > start && atomicLoad(&adjacency[entries(j-1u)]) > value {
            atomicStore(&adjacency[entries(j)],atomicLoad(&adjacency[entries(j-1u)]));
            j--;
        }
        atomicStore(&adjacency[entries(j)],value);
    }
}

@compute @workgroup_size(1)
fn reset_evaluation() { atomicStore(&status[1],atomicLoad(&status[0])); }

@compute @workgroup_size(64)
fn rigid_partials(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let r = group.x*64u+lane;
    let d = group.y;
    var sum = vec2<f32>(0.0);
    if r < u.rows && enabled(contacts[r].ids.w) {
        let change = impulses[r].xyz-rows[r].lambda.xyz;
        let direction = impulse_direction[r].xyz;
        for (var k = 0u; k < 3u; k++) {
            let weight = response[(r*3u+k)*u.dofs+d];
            sum += weight*vec2<f32>(change[k],direction[k]);
        }
    }
    reductions[lane] = sum;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride { reductions[lane] += reductions[lane+stride]; }
        workgroupBarrier();
    }
    if lane == 0u { partials[group.x*u.dofs+d] = reductions[0]; }
}

@compute @workgroup_size(64)
fn rigid_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if d >= u.dofs { return; }
    var sum = vec2<f32>(0.0);
    for (var block = 0u; block < (u.rows+63u)/64u; block++) { sum += partials[block*u.dofs+d]; }
    let residual = rigid_velocity[d]-rigid_reference[d]+sum.x;
    let action = rigid_direction[d]+sum.y;
    if !finite(residual) || !finite(action) { atomicOr(&status[1],4u); }
    rigid_terms[d] = vec4<f32>(residual,action,sum);
}

@compute @workgroup_size(64)
fn soft_transpose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node = gid.x;
    if node >= u.particles || atomicLoad(&status[0]) != 0u { return; }
    var sum = vec3<f32>(0.0);
    var direction = vec3<f32>(0.0);
    var sum_error = vec3<f32>(0.0);
    var direction_error = vec3<f32>(0.0);
    let start = atomicLoad(&adjacency[offsets(node)]);
    let end = atomicLoad(&adjacency[offsets(node+1u)]);
    for (var slot = start; slot < end; slot++) {
        let item = atomicLoad(&adjacency[entries(slot)]);
        let r = item/3u;
        let c = contacts[r];
        let axes = basis(c.normal_gap.xyz);
        let weight = c.weights[item%3u];
        let term = weight*(axes*(impulses[r].xyz-rows[r].lambda.xyz));
        let dterm = weight*(axes*impulse_direction[r].xyz);
        let next_sum = sum+term;
        let next_direction = direction+dterm;
        sum_error += select((term-next_sum)+sum,(sum-next_sum)+term,abs(sum)>=abs(term));
        direction_error += select((dterm-next_direction)+direction,(direction-next_direction)+dterm,abs(direction)>=abs(dterm));
        sum = next_sum;
        direction = next_direction;
    }
    sum = (sum+sum_error)*inverse_mass[node];
    direction = (direction+direction_error)*inverse_mass[node];
    if !finite3(sum) || !finite3(direction) || !finite3(soft_velocity[node].xyz)
        || !finite3(soft_direction[node].xyz) { atomicOr(&status[1],4u); }
    soft_terms[node] = vec4<f32>(sum,0.0);
    soft_terms[u.particles+node] = vec4<f32>(direction,0.0);
}

@compute @workgroup_size(64)
fn contact_evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= u.rows || atomicLoad(&status[0]) != 0u { return; }
    let c = contacts[r];
    let lambda = impulses[r].xyz;
    let dlambda = impulse_direction[r].xyz;
    var value = lambda;
    var action = dlambda;
    if enabled(c.ids.w) {
        var relative = vec3<f32>(0.0);
        var drelative = vec3<f32>(0.0);
        if c.ids.w == 1u {
            let axes = transpose(basis(c.normal_gap.xyz));
            for (var i = 0u; i < c.support.z; i++) {
                relative += c.weights[i]*(axes*soft_velocity[node_at(c,i)].xyz);
                drelative += c.weights[i]*(axes*soft_direction[node_at(c,i)].xyz);
            }
        }
        for (var k = 0u; k < 3u; k++) {
            for (var d = 0u; d < u.dofs; d++) {
                let weight = projected[(r*3u+k)*u.dofs+d];
                relative[k] -= weight*rigid_velocity[d];
                drelative[k] -= weight*rigid_direction[d];
            }
        }
        let row = rows[r];
        if c.ids.w == 3u {
            let desired = loop_target(c);
            let diagonal = vec3<f32>(row.normal,row.tangent,row.target_speed);
            for (var k = 0u; k < 3u; k++) {
                if diagonal[k] > 0.0 {
                    value[k] = relative[k]-desired[k];
                    action[k] = drelative[k];
                } else if relative[k] != desired[k] || drelative[k] != 0.0 {
                    atomicOr(&status[1],2u);
                }
            }
        } else {
            let trial_normal = lambda.x+(row.target_speed-relative.x)/row.normal;
            value.x = select(lambda.x*row.normal,relative.x-row.target_speed,trial_normal > 0.0);
            action.x = select(row.normal*dlambda.x,drelative.x,trial_normal > 0.0);
            if c.ids.w == 1u {
                let trial = lambda.yz-relative.yz/row.tangent;
                let dtrial = dlambda.yz-drelative.yz/row.tangent;
                let radius = c.material.x*max(0.0,lambda.x);
                let dradius = c.material.x*select(0.0,dlambda.x,lambda.x > 0.0);
                let magnitude = length(trial);
                if radius == 0.0 && magnitude == 0.0 {
                    value.y = lambda.y*row.tangent;
                    value.z = lambda.z*row.tangent;
                    action.y = dlambda.y*row.tangent;
                    action.z = dlambda.z*row.tangent;
                } else if magnitude > radius {
                    let axis = trial/magnitude;
                    let projection = radius*axis;
                    let derivative = (radius/magnitude)*(dtrial-axis*dot(axis,dtrial))+dradius*axis;
                    value.y = (lambda.y-projection.x)*row.tangent;
                    value.z = (lambda.z-projection.y)*row.tangent;
                    action.y = (dlambda.y-derivative.x)*row.tangent;
                    action.z = (dlambda.z-derivative.y)*row.tangent;
                } else {
                    value.y = relative.y;
                    value.z = relative.z;
                    action.y = drelative.y;
                    action.z = drelative.z;
                }
            }
        }
    }
    if !finite3(value) || !finite3(action) || !finite3(lambda) || !finite3(dlambda) {
        atomicOr(&status[1],4u);
    }
    contact_terms[r] = vec4<f32>(value,0.0);
    contact_terms[u.rows+r] = vec4<f32>(action,0.0);
}
