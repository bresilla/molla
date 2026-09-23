override SWEEP_LANES: u32 = 64u;
override SWEEP_DOFS: u32 = 64u;
var<workgroup> sweep_velocity: array<f32, 64>;
var<workgroup> sweep_projected: array<array<f32, 64>, 3>;
var<workgroup> sweep_response: array<array<f32, 64>, 3>;
var<workgroup> sweep_count: u32;
var<workgroup> sweep_fault: u32;

fn cooperative_relative(contact: Contact, direction: vec3<f32>, k: u32) -> f32 {
    var value = 0.0;
    for (var i = 0u; i < contact.support.z; i += 1u) {
        value += contact.weights[i] * dot(velocities[support_node(contact, i)].xyz, direction);
    }
    if contact.ids.y != 0xffffffffu {
        for (var d = 0u; d < SWEEP_DOFS; d += 1u) {
            value -= sweep_projected[k][d] * sweep_velocity[d];
        }
    }
    return value;
}

fn cooperative_delta(contact: Contact, direction: vec3<f32>, k: u32, impulse: f32) {
    for (var i = 0u; i < contact.support.z; i += 1u) {
        let particle = support_node(contact, i);
        velocities[particle] = vec4<f32>(velocities[particle].xyz + direction * (impulse*contact.weights[i]*inverse_mass[particle]), 0.0);
    }
    if contact.ids.y != 0xffffffffu {
        cooperative_rigid_delta(k, impulse);
    }
}

fn cooperative_row(r: u32) -> u32 {
    let contact = contacts[r];
    if contact.ids.w == 3u {
        let row = rows[r];
        let diagonal = vec3<f32>(row.inverse_normal, row.tangent_lipschitz, row.target_speed);
        let desired = loop_target(contact);
        var lambda = row.lambda.xyz;
        for (var k = 0u; k < 3u; k += 1u) {
            let residual = desired[k] - cooperative_relative(contact, vec3<f32>(0.0), k);
            if !finite(residual) { return 4u; }
            if diagonal[k] == 0.0 {
                if residual != 0.0 { return 2u; }
                continue;
            }
            let impulse = residual / diagonal[k];
            if !finite(impulse) { return 4u; }
            cooperative_delta(contact, vec3<f32>(0.0), k, impulse);
            lambda[k] += impulse;
        }
        if !finite(dot(lambda,lambda)) { return 4u; }
        rows[r].lambda = vec4<f32>(lambda,0.0);
        return 0u;
    }
    if contact.ids.w != 1u { return 0u; }
    let directions = basis(contact.normal_gap.xyz);
    let row = rows[r];
    var lambda = row.lambda.xyz;
    if row.inverse_normal > 0.0 {
        lambda.x = max(0.0, lambda.x + (row.target_speed-cooperative_relative(contact,directions[0],0u))/row.inverse_normal);
    }
    cooperative_delta(contact, directions[0], 0u, lambda.x-row.lambda.x);
    if row.tangent_lipschitz > 0.0 {
        lambda.y -= cooperative_relative(contact,directions[1],1u)/row.tangent_lipschitz;
        lambda.z -= cooperative_relative(contact,directions[2],2u)/row.tangent_lipschitz;
    }
    let tangent = length(lambda.yz);
    let limit = contact.material.x*lambda.x;
    if tangent > limit { lambda.y *= limit/tangent; lambda.z *= limit/tangent; }
    cooperative_delta(contact, directions[1], 1u, lambda.y-row.lambda.y);
    cooperative_delta(contact, directions[2], 2u, lambda.z-row.lambda.z);
    rows[r].lambda = vec4<f32>(lambda,0.0);
    if !finite(dot(lambda,lambda)) { return 4u; }
    return 0u;
}

@compute @workgroup_size(SWEEP_LANES)
fn solve_cooperative(@builtin(local_invocation_index) lane: u32) {
    if lane == 0u {
        sweep_count = active_rows[0];
        sweep_fault = atomicLoad(&status);
    }
    let count = workgroupUniformLoad(&sweep_count);
    let initial_fault = workgroupUniformLoad(&sweep_fault);
    if initial_fault != 0u { return; }
    if lane < SWEEP_DOFS { sweep_velocity[lane] = rigid_velocity[lane]; }
    workgroupBarrier();
    for (var iteration = 0u; iteration < params.iterations; iteration += 1u) {
        for (var rank = 0u; rank < count; rank += 1u) {
            let r = active_rows[rank+1u];
            if lane < SWEEP_DOFS && contacts[r].ids.y != 0xffffffffu {
                for (var k = 0u; k < 3u; k += 1u) {
                    sweep_projected[k][lane] = projected[(r*3u+k)*SWEEP_DOFS+lane];
                    sweep_response[k][lane] = response[(r*3u+k)*SWEEP_DOFS+lane];
                }
            }
            workgroupBarrier();
            if lane == 0u && sweep_fault == 0u { sweep_fault = cooperative_row(r); }
            workgroupBarrier();
        }
    }
    if lane < SWEEP_DOFS { rigid_velocity[lane] = sweep_velocity[lane]; }
    if lane == 0u && sweep_fault != 0u { atomicOr(&status, sweep_fault); }
}
