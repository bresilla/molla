@group(6) @binding(0) var<storage, read> material_direction: array<vec4<f32>>;
@group(6) @binding(1) var<storage, read_write> material_residual: array<vec4<f32>>;
@group(6) @binding(2) var<storage, read_write> material_action: array<vec4<f32>>;
struct MaterialTangent { columns: array<mat3x3<f32>,9> }
@group(6) @binding(3) var<storage, read_write> material_tangents: array<MaterialTangent>;

fn material_free_direction(node: u32) -> vec3<f32> {
    if particle_inv_mass[node] <= 0.0 { return vec3<f32>(0.0); }
    return material_direction[node].xyz;
}

fn material_stress_action(tet: u32, previous: mat3x3<f32>, increment: mat3x3<f32>, df: mat3x3<f32>) -> mat3x3<f32> {
    let f = previous + increment;
    let cof = cofactor3(f);
    var dcof: mat3x3<f32>;
    for (var column = 0u; column < 3u; column++) {
        let b = (column+1u)%3u;
        let c = (column+2u)%3u;
        dcof[column] = cross(df[b],f[c]) + cross(f[b],df[c]);
    }
    let dj = dot(df[0],cof[0]) + dot(f[0],dcof[0]);
    let j_minus_one = tet_dot_minus_one(f[0],cof[0]);
    let mu = tet_mu[tet];
    let lambda = tet_lambda[tet] + mu;
    var dp = (df-dcof)*mu + (cof*dj+dcof*j_minus_one)*lambda;
    let viscosity = tet_viscosity[tet];
    if any(viscosity > vec2<f32>(0.0)) {
        let strain = (transpose(previous)*increment + transpose(increment)*previous
            + transpose(increment)*increment)*0.5;
        let dstrain = (transpose(df)*f + transpose(f)*df)*0.5;
        let trace = strain[0].x+strain[1].y+strain[2].z;
        let dtrace = dstrain[0].x+dstrain[1].y+dstrain[2].z;
        let stress = (strain-IDENTITY3*(trace/3.0))*(2.0*viscosity.x)
            + IDENTITY3*(viscosity.y*trace);
        let dstress = (dstrain-IDENTITY3*(dtrace/3.0))*(2.0*viscosity.x)
            + IDENTITY3*(viscosity.y*dtrace);
        dp += (df*stress+f*dstress)*(1.0/u.dt);
    }
    return dp;
}

fn material_tet_action(tet: u32, order: u32) -> vec3<f32> {
    let plastic = tet_f_plastic[tet];
    if any(plastic.col0.xyz != IDENTITY3[0]) || any(plastic.col1.xyz != IDENTITY3[1])
        || any(plastic.col2.xyz != IDENTITY3[2]) {
        return vec3<f32>(bitcast<f32>(0x7fc00000u));
    }
    let origin = tet_indices[tet*4u];
    let raw = tet_dm_inv[tet];
    let inverse = mat3x3<f32>(raw.col0.xyz, raw.col1.xyz, raw.col2.xyz);
    var edges: mat3x3<f32>;
    var direction_edges: mat3x3<f32>;
    for (var column = 0u; column < 3u; column++) {
        let node = tet_indices[tet*4u+column+1u];
        edges[column] = tet_displacement_edge(origin, node);
        direction_edges[column] = material_free_direction(node) - material_free_direction(origin);
    }
    let previous_parts = tet_previous_parts(tet, edges, inverse);
    let increment_parts = tet_increment_parts(tet, inverse);
    let dp = material_stress_action(tet, previous_parts.value + previous_parts.correction,
        increment_parts.value + increment_parts.correction, direction_edges * inverse);
    var gradient: vec3<f32>;
    if order == 0u {
        gradient = -vec3<f32>(dot(inverse[0],vec3<f32>(1.0)),
            dot(inverse[1],vec3<f32>(1.0)),dot(inverse[2],vec3<f32>(1.0)));
    } else {
        gradient = vec3<f32>(inverse[0][order-1u],inverse[1][order-1u],inverse[2][order-1u]);
    }
    return -(dp*gradient)*tet_rest_volume[tet];
}

@compute @workgroup_size(64)
fn cache_material_tangents(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.work_count { return; }
    let tet = u.color_index + gid.x;
    let origin = tet_indices[tet*4u];
    let raw = tet_dm_inv[tet];
    let inverse = mat3x3<f32>(raw.col0.xyz,raw.col1.xyz,raw.col2.xyz);
    var edges: mat3x3<f32>;
    for (var c = 0u; c < 3u; c++) { edges[c] = tet_displacement_edge(origin,tet_indices[tet*4u+c+1u]); }
    let previous = tet_previous_parts(tet,edges,inverse);
    let increment = tet_increment_parts(tet,inverse);
    let plastic = tet_f_plastic[tet];
    let valid = all(plastic.col0.xyz == IDENTITY3[0]) && all(plastic.col1.xyz == IDENTITY3[1])
        && all(plastic.col2.xyz == IDENTITY3[2]);
    for (var c = 0u; c < 9u; c++) {
        var df = mat3x3<f32>();
        df[c/3u][c%3u] = 1.0;
        var value = material_stress_action(tet,previous.value+previous.correction,
            increment.value+increment.correction,df);
        if !valid {
            let invalid = vec3<f32>(bitcast<f32>(0x7fc00000u));
            value = mat3x3<f32>(invalid,invalid,invalid);
        }
        material_tangents[tet].columns[c] = value;
    }
}

fn cached_tet_action(tet: u32, order: u32) -> vec3<f32> {
    let origin = tet_indices[tet*4u];
    let raw = tet_dm_inv[tet];
    let inverse = mat3x3<f32>(raw.col0.xyz,raw.col1.xyz,raw.col2.xyz);
    var edges: mat3x3<f32>;
    for (var c = 0u; c < 3u; c++) {
        edges[c] = material_free_direction(tet_indices[tet*4u+c+1u])-material_free_direction(origin);
    }
    let df = edges*inverse;
    var dp = mat3x3<f32>();
    for (var c = 0u; c < 9u; c++) { dp += material_tangents[tet].columns[c]*df[c/3u][c%3u]; }
    var gradient: vec3<f32>;
    if order == 0u {
        gradient = -vec3<f32>(dot(inverse[0],vec3<f32>(1.0)),dot(inverse[1],vec3<f32>(1.0)),dot(inverse[2],vec3<f32>(1.0)));
    } else { gradient = vec3<f32>(inverse[0][order-1u],inverse[1][order-1u],inverse[2][order-1u]); }
    return -(dp*gradient)*tet_rest_volume[tet];
}

@compute @workgroup_size(64)
fn material_cached_action(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node = gid.x;
    if node >= u.particle_count { return; }
    let inverse_mass = particle_inv_mass[node];
    let direction = material_direction[node].xyz;
    if inverse_mass <= 0.0 { material_action[node] = vec4<f32>(direction,0.0); return; }
    var derivative = vec3<f32>(0.0);
    for (var s = vertex_adj_offsets[node]; s < vertex_adj_offsets[node+1u]; s++) {
        derivative += material_spring_action(node,vertex_adj_springs[s]);
    }
    for (var t = vertex_adj_tets_offsets[node]; t < vertex_adj_tets_offsets[node+1u]; t += 2u) {
        derivative += cached_tet_action(vertex_adj_tets[t],vertex_adj_tets[t+1u]);
    }
    material_action[node] = vec4<f32>(direction-(u.dt*u.dt*inverse_mass)*derivative,0.0);
}

fn material_spring_action(node: u32, spring: u32) -> vec3<f32> {
    let other = select(spring_a[spring],spring_b[spring],spring_a[spring] == node);
    let previous = particle_q_in[node].xyz-particle_q_in[other].xyz;
    let increment = material_displacement(node)-material_displacement(other);
    let delta = previous+increment;
    let size = max(length(delta),1e-8);
    let rest = spring_rest_length[spring];
    var extension = size-rest;
    if length(delta) >= 1e-8 {
        let low = tet_sum_error(particle_q_in[node].xyz,-particle_q_in[other].xyz,previous)
            + tet_sum_error(material_displacement(node),-material_displacement(other),increment)
            + tet_sum_error(previous,increment,delta);
        extension = spring_length_extension(delta,low,rest,size);
    }
    if spring_tension_only[spring] != 0u && extension <= 0.0 { return vec3<f32>(0.0); }
    let direction = material_free_direction(node)-material_free_direction(other);
    if length(delta) < 1e-8 { return -spring_stiffness[spring]*(extension/size)*direction; }
    let axis = delta/size;
    return -spring_stiffness[spring]*((extension/size)*direction
        + (rest/size)*dot(axis,direction)*axis);
}

@compute @workgroup_size(64)
fn material_linearization(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node = u.color_index + gid.x;
    if node >= u.particle_count { return; }
    let inverse_mass = particle_inv_mass[node];
    let direction = material_direction[node].xyz;
    if inverse_mass <= 0.0 {
        material_residual[node] = vec4<f32>(particle_qd[node].xyz,0.0);
        material_action[node] = vec4<f32>(direction,0.0);
        return;
    }
    var force = vec3<f32>(0.0);
    var derivative = vec3<f32>(0.0);
    for (var s = vertex_adj_offsets[node]; s < vertex_adj_offsets[node+1u]; s++) {
        let spring = vertex_adj_springs[s];
        var f: vec3<f32>;
        var h: mat3x3<f32>;
        spring_force_hessian(node,spring,true,&f,&h);
        force += f;
        derivative += material_spring_action(node,spring);
    }
    for (var t = vertex_adj_tets_offsets[node]; t < vertex_adj_tets_offsets[node+1u]; t += 2u) {
        let tet = vertex_adj_tets[t];
        let order = vertex_adj_tets[t+1u];
        var f: vec3<f32>;
        var h: mat3x3<f32>;
        tet_neo_hookean_block(tet,order,true,&f,&h);
        force += f;
        derivative += material_tet_action(tet,order);
    }
    material_residual[node] = vec4<f32>((particle_delta[node].xyz-inertia_y[node].xyz)/u.dt
        - (u.dt*inverse_mass)*force,0.0);
    material_action[node] = vec4<f32>(direction-(u.dt*u.dt*inverse_mass)*derivative,0.0);
}

@compute @workgroup_size(64)
fn material_inverse_diagonal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node = u.color_index + gid.x;
    if node >= u.particle_count { return; }
    let inverse_mass = particle_inv_mass[node];
    var diagonal = vec3<f32>(0.0);
    if inverse_mass > 0.0 {
        for (var s = vertex_adj_offsets[node]; s < vertex_adj_offsets[node+1u]; s++) {
            var force: vec3<f32>;
            var hessian: mat3x3<f32>;
            spring_force_hessian(node,vertex_adj_springs[s],true,&force,&hessian);
            diagonal += vec3<f32>(hessian[0].x,hessian[1].y,hessian[2].z);
        }
        for (var t = vertex_adj_tets_offsets[node]; t < vertex_adj_tets_offsets[node+1u]; t += 2u) {
            var force: vec3<f32>;
            var hessian: mat3x3<f32>;
            tet_neo_hookean_block(vertex_adj_tets[t],vertex_adj_tets[t+1u],true,&force,&hessian);
            diagonal += vec3<f32>(hessian[0].x,hessian[1].y,hessian[2].z);
        }
    }
    let scaled = vec3<f32>(1.0)+(u.dt*u.dt*inverse_mass)*diagonal;
    if !all(abs(scaled) <= vec3<f32>(3.402823e38)) {
        material_action[node] = vec4<f32>(bitcast<f32>(0x7fc00000u));
        return;
    }
    material_action[node] = vec4<f32>(vec3<f32>(1.0)/max(vec3<f32>(1.0),scaled),0.0);
}
