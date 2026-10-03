struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct Body { com: vec4<f32>, response: vec4<u32> }
struct Acceleration { linear: vec4<f32>, angular: vec4<f32> }
struct Art { first: u32, width: u32, mass: u32, projected: u32 }
struct Row { point: vec4<f32>, normal: vec4<f32>, coefficients: vec4<f32>, endpoints: vec4<u32>,
    relative: vec4<f32>, surface: vec4<f32>, force: vec4<f32>, penalty: vec4<f32> }
struct Solution { impulse: vec2<f32>, slip: vec2<f32>, cap: f32, regime: u32, iterations: u32, residual: f32 }
struct ScalarRow { endpoints: vec4<u32>, velocity: vec4<f32> }
struct Params { arts: u32, bodies: u32, contacts: u32, islands: u32, rows: u32, dofs: u32,
    projected: u32, whitened: u32, scales: u32, scalars: u32, retained_delta: u32, impact: u32 }
@group(0) @binding(0) var<storage, read> rows: array<Row>;
@group(0) @binding(1) var<storage, read> bodies: array<Body>;
@group(0) @binding(2) var<storage, read> arts: array<Art>;
@group(0) @binding(3) var<storage, read> poses: array<Transform>;
@group(0) @binding(4) var<storage, read> masses: array<f32>;
@group(0) @binding(5) var<storage, read> jacobians: array<f32>;
@group(0) @binding(6) var<storage, read> input_rates: array<f32>;
@group(0) @binding(7) var<storage, read> predicted: array<f32>;
@group(0) @binding(8) var<storage, read> island_count: array<u32>;
@group(0) @binding(9) var<storage, read> counts: array<u32>;
@group(0) @binding(10) var<storage, read> indices: array<u32>;
@group(0) @binding(11) var<storage, read> art_islands: array<u32>;
@group(0) @binding(12) var<storage, read> step_size: array<f32>;
@group(0) @binding(13) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(14) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(15) var<storage, read_write> gram: array<f32>;
@group(0) @binding(16) var<storage, read_write> free: array<f32>;
@group(0) @binding(17) var<storage, read_write> disk_input: array<u32>;
@group(0) @binding(18) var<storage, read_write> directions: array<vec4<f32>>;
@group(0) @binding(19) var<storage, read> solutions: array<u32>;
@group(0) @binding(20) var<storage, read_write> delta: array<atomic<u32>>;
@group(0) @binding(21) var<storage, read_write> corrected: array<f32>;
@group(0) @binding(22) var<storage, read_write> world_impulses: array<vec4<f32>>;
@group(0) @binding(23) var<uniform> u: Params;
@group(0) @binding(24) var<storage, read> art_offsets: array<u32>;
@group(0) @binding(25) var<storage, read> art_indices: array<u32>;
@group(0) @binding(26) var<storage, read> scalar_rows: array<ScalarRow>;
@group(0) @binding(27) var<storage, read> scalar_jacobians: array<f32>;
@group(0) @binding(28) var<storage, read_write> block_kinds: array<u32>;
@group(0) @binding(29) var<storage, read_write> scalar_solutions: array<Solution>;
@group(0) @binding(30) var<storage, read> predictor_delta: array<f32>;
@group(0) @binding(31) var<storage, read> boundary_acceleration: array<Acceleration>;
@group(0) @binding(32) var<storage, read_write> predicted_contacts: array<vec2<u32>>;
const NONE: u32 = 0xffffffffu;
const INPUT: u32 = 4194304u;
const MASS: u32 = 8388608u;
const NONFINITE: u32 = 16777216u;

fn width() -> u32 { return select(2u, 3u, u.impact != 0u); }
fn data(slot: u32, field: u32) -> f32 {
    return bitcast<f32>(disk_input[slot*select(4u,8u,u.impact != 0u)+field]);
}
fn set_data(slot: u32, field: u32, value: f32) {
    disk_input[slot*select(4u,8u,u.impact != 0u)+field] = bitcast<u32>(value);
}
fn impulse(slot: u32, axis: u32) -> f32 {
    return bitcast<f32>(solutions[slot*select(8u,12u,u.impact != 0u)+axis]);
}
fn impulse_residue(slot: u32, axis: u32) -> f32 {
    if u.impact!=0u || block_kinds[slot]!=0u { return 0.0; }
    return bitcast<f32>(solutions[8u*u.islands*u.contacts+2u*slot+axis]);
}
fn scalar_solution(slot: u32) -> Solution {
    if u.impact != 0u {
        let start = slot*12u;
        return Solution(vec2<f32>(impulse(slot,0u),0.0),vec2<f32>(bitcast<f32>(solutions[start+4u]),0.0),
            0.0,3u+block_kinds[slot],solutions[start+10u],bitcast<f32>(solutions[start+11u]));
    }
    let start = slot*8u;
    return Solution(vec2<f32>(impulse(slot,0u),impulse(slot,1u)),
        vec2<f32>(bitcast<f32>(solutions[start+2u]),bitcast<f32>(solutions[start+3u])),
        bitcast<f32>(solutions[start+4u]),solutions[start+5u],solutions[start+6u],bitcast<f32>(solutions[start+7u]));
}

fn finite(value: f32) -> bool { return value == value && abs(value) <= 3.4028234e38; }
fn response_round(index: u32, value: f32) -> f32 {
    atomicStore(&delta[index],bitcast<u32>(value));
    return bitcast<f32>(atomicLoad(&delta[index]));
}
fn response_sum(index: u32, a: f32, b: f32) -> vec2<f32> {
    let high=response_round(index,a+b);
    let virtual_b=response_round(index,high-a);
    let virtual_a=response_round(index,high-virtual_b);
    let error_a=response_round(index,a-virtual_a);
    let error_b=response_round(index,b-virtual_b);
    return vec2<f32>(high,response_round(index,error_a+error_b));
}
fn response_add(index: u32, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let sum=response_sum(index,a.x,b.x);
    let low=response_round(index,response_round(index,sum.y+a.y)+b.y);
    return response_sum(index,sum.x,low);
}
fn response_product(index: u32, weight: f32, value: vec2<f32>) -> vec2<f32> {
    let high=response_round(index,weight*value.x);
    return vec2<f32>(high,fma(weight,value.x,-high)+weight*value.y);
}
fn response_divide(index: u32, value: vec2<f32>, divisor: f32) -> vec2<f32> {
    let high=response_round(index,value.x/divisor);
    return response_sum(index,high,(fma(-high,divisor,value.x)+value.y)/divisor);
}
fn finite3(value: vec3<f32>) -> bool { return finite(value.x) && finite(value.y) && finite(value.z); }
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
fn owner(body: u32) -> u32 {
    if (body == NONE) { return NONE; }
    if (body >= u.bodies) { atomicOr(&status, INPUT); return NONE; }
    return bodies[body].response.x;
}
fn boundary_change(body: u32, point: vec3<f32>) -> vec3<f32> {
    if body == NONE { return vec3<f32>(0.0); }
    if body >= u.bodies { atomicOr(&status, INPUT); return vec3<f32>(0.0); }
    let boundary = bodies[body].response.w;
    if boundary == NONE { return vec3<f32>(0.0); }
    let center = poses[boundary].position.xyz + rotate(poses[boundary].rotation, bodies[boundary].com.xyz);
    let acceleration = boundary_acceleration[boundary];
    return (acceleration.linear.xyz + cross(acceleration.angular.xyz, point - center)) * step_size[0];
}
fn column(body: u32, art: u32, point: vec3<f32>, direction: vec3<f32>, k: u32) -> f32 {
    if (body == NONE) { return 0.0; }
    if (body >= u.bodies) { atomicOr(&status, INPUT); return 0.0; }
    let body_meta = bodies[body];
    if (body_meta.response.x != art) { return 0.0; }
    let offset = body_meta.response.y + k; let n = body_meta.response.z;
    let center = poses[body].position.xyz + rotate(poses[body].rotation, body_meta.com.xyz);
    let torque = cross(point - center, direction);
    return dot(direction, vec3<f32>(jacobians[offset], jacobians[offset+n], jacobians[offset+2u*n]))
        + dot(torque, vec3<f32>(jacobians[offset+3u*n], jacobians[offset+4u*n], jacobians[offset+5u*n]));
}

@compute @workgroup_size(64)
fn predict_contacts(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= u.rows || atomicLoad(&status) != 0u) { return; }
    let row = rows[i];
    if (row.coefficients.w == 0.0) { return; }
    let dt = step_size[0];
    if (!finite(dt) || dt <= 0.0 || row.coefficients.w != 1.0
        || abs(dot(row.normal.xyz,row.normal.xyz)-1.0) > 1e-3) { atomicOr(&status, INPUT); return; }
    if (!finite3(row.normal.xyz) || !finite3(row.point.xyz) || !finite3(row.relative.xyz)) {
        atomicOr(&status, NONFINITE); return;
    }
    if (u.impact == 3u) {
        let onset = bitcast<u32>(predictor_delta[i]);
        if (onset > 1u || !finite(row.point.w) || row.point.w < 0.0) { atomicOr(&status, INPUT); return; }
        predicted_contacts[i] = vec2<u32>(u32(onset != 0u && row.point.w > 0.0),bitcast<u32>(row.point.w));
        return;
    }
    if (!finite3(row.coefficients.xyz) || !finite3(row.surface.xyz)) { atomicOr(&status, NONFINITE); return; }
    if (row.coefficients.y < 0.0 || row.coefficients.x < row.coefficients.y
        || row.coefficients.z < 0.0 || row.coefficients.z > 1.0) { atomicOr(&status, INPUT); return; }
    let a = owner(row.endpoints.z); let b = owner(row.endpoints.w);
    let first = min(a,b); let last = max(a,b);
    var speed = dot(row.normal.xyz,row.relative.xyz);
    speed += dot(row.normal.xyz, boundary_change(row.endpoints.w,row.point.xyz)
        - boundary_change(row.endpoints.z,row.point.xyz));
    for (var side = 0u; side < 2u; side += 1u) {
        let owner = select(first,last,side == 1u);
        if (owner == NONE || (side == 1u && first == last)) { continue; }
        let art = arts[owner];
        for (var k = 0u; k < art.width; k += 1u) {
            let projected = column(row.endpoints.w,owner,row.point.xyz,row.normal.xyz,k)
                - column(row.endpoints.z,owner,row.point.xyz,row.normal.xyz,k);
            var change = predicted[art.first+k] - input_rates[art.first+k];
            if (u.retained_delta & 1u) != 0u { change = predictor_delta[art.first+k]; }
            speed += projected * change;
        }
    }
    if (!finite(speed)) { atomicOr(&status, NONFINITE); return; }
    predicted_contacts[i] = vec2<u32>(u32(speed <= 0.0),bitcast<u32>(speed));
}

@compute @workgroup_size(64)
fn geometry(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = id.x;
    if (atomicLoad(&status) != 0u) { return; }
    let dt = step_size[0];
    if (!finite(dt) || dt <= 0.0 || island_count[0] > u.islands) { atomicOr(&status, INPUT); return; }
    let island = slot / u.contacts; let c = slot % u.contacts;
    if (island >= island_count[0]) { return; }
    if (counts[island] > u.contacts) { atomicOr(&status, INPUT); return; }
    if (c >= counts[island]) { return; }
    let index = indices[slot];
    if (index >= u.rows + u.scalars) { atomicOr(&status, INPUT); return; }
    if (index >= u.rows) {
        let scalar = index - u.rows;
        let row = scalar_rows[scalar];
        let a = owner(row.endpoints.x); let b = owner(row.endpoints.y);
        let bounded = row.endpoints.z == 3u;
        if (row.endpoints.w != 1u || row.endpoints.z < 1u || row.endpoints.z > 3u
            || (!bounded && any(row.velocity.yzw != vec3<f32>(0.0)))) { atomicOr(&status, INPUT); return; }
        if bounded && u.impact == 3u { atomicOr(&status, INPUT); return; }
        if (!finite(row.velocity.x)) { atomicOr(&status, NONFINITE); return; }
        for (var member = 0u; member < u.arts; member += 1u) {
            let art = arts[member];
            for (var k = 0u; k < art.width; k += 1u) {
                let value = scalar_jacobians[scalar*u.dofs+art.first+k];
                if (!finite(value)) { atomicOr(&status, NONFINITE); return; }
                if (member != a && member != b && value != 0.0) { atomicOr(&status, INPUT); return; }
            }
        }
        block_kinds[slot] = row.endpoints.z;
        free[slot*width()] = row.velocity.x;
        if u.impact != 0u {
            for (var lane = 0u; lane < 4u; lane += 1u) { disk_input[slot*8u+lane] = 0u; }
            if bounded {
                set_data(slot,0u,row.velocity.z);
                set_data(slot,1u,row.velocity.w);
                set_data(slot,2u,row.velocity.y);
            }
            disk_input[slot*8u+4u] = row.endpoints.z;
        } else {
            for (var lane = 0u; lane < 4u; lane += 1u) { disk_input[slot*4u+lane] = 0u; }
            if bounded {
                set_data(slot,0u,row.velocity.z);
                set_data(slot,1u,row.velocity.w);
                set_data(slot,2u,row.velocity.y);
            }
        }
        return;
    }
    let row = rows[index];
    let normal = row.normal.xyz;
    if (!finite3(row.point.xyz) || !finite3(normal) || !finite3(row.relative.xyz) || !finite3(row.surface.xyz)) { atomicOr(&status, NONFINITE); return; }
    let squared = dot(normal, normal);
    if (abs(squared - 1.0) > 1e-3) { atomicOr(&status, INPUT); return; }
    let basis = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.x) < 0.6);
    let first = normalize(cross(normal, basis)); let second = cross(normal, first);
    let slip = row.relative.xyz + row.surface.xyz;
    if u.impact != 0u {
        directions[3u*slot] = vec4<f32>(normal,0.0);
        if u.impact == 3u {
            free[3u*slot] = -row.point.w;
            block_kinds[slot] = 2u;
            disk_input[slot*8u+4u] = 2u;
            return;
        }
        directions[3u*slot+1u] = vec4<f32>(first,0.0);
        directions[3u*slot+2u] = vec4<f32>(second,0.0);
        for (var i = 0u; i < 3u; i += 1u) { set_data(slot,i,row.coefficients[i]); }
        free[3u*slot] = dot(normal,row.relative.xyz);
        free[3u*slot+1u] = dot(first,slip);
        free[3u*slot+2u] = dot(second,slip);
        return;
    }
    directions[2u * slot] = vec4<f32>(first, 0.0); directions[2u * slot + 1u] = vec4<f32>(second, 0.0);
    let cap = row.coefficients.xy * (row.normal.w * dt);
    if (!finite(cap.x) || !finite(cap.y) || (row.coefficients.x > 0.0 && cap.x < 1.17549435e-38)
        || (row.coefficients.y > 0.0 && cap.y < 1.17549435e-38)) { atomicOr(&status, NONFINITE); return; }
    set_data(slot,0u,cap.x); set_data(slot,1u,cap.y);
    set_data(slot,2u,dot(first,slip)); set_data(slot,3u,dot(second,slip));
}

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) id: vec3<u32>) {
    let art_index = id.x;
    if (atomicLoad(&status) != 0u || art_index >= u.arts) { return; }
    let island = art_islands[art_index];
    if (island == NONE) { return; }
    if (island >= island_count[0] || counts[island] > u.contacts) { atomicOr(&status, INPUT); return; }
    let art = arts[art_index]; let n = art.width; let f = art.mass;
    var scale = 0.0;
    for (var i = 0u; i < n*n; i += 1u) {
        let value = masses[f+i];
        if (!finite(value)) { atomicOr(&status, NONFINITE); return; }
        scale = max(scale, abs(value));
    }
    let root_scale = sqrt(scale);
    scratch[u.scales + art_index] = root_scale;
    for (var i = 0u; i < n; i += 1u) {
        for (var j = 0u; j <= i; j += 1u) {
            var value = 0.0;
            if (scale > 0.0) { value = masses[f+i*n+j] / scale; }
            for (var k = 0u; k < j; k += 1u) { value -= scratch[f+i*n+k] * scratch[f+j*n+k]; }
            if (!finite(value)) { atomicOr(&status, NONFINITE); return; }
            if (i == j) {
                if (value <= 1e-6 && masses[f+i*n+i] != 0.0) { atomicOr(&status, MASS); return; }
                if (value < -1e-6) { atomicOr(&status, MASS); return; }
                scratch[f+i*n+i] = sqrt(max(value, 0.0));
            } else {
                let diagonal = scratch[f+j*n+j];
                if (diagonal == 0.0) {
                    if (abs(value) > 1e-6) { atomicOr(&status, MASS); return; }
                    scratch[f+i*n+j] = 0.0;
                } else { scratch[f+i*n+j] = value / diagonal; }
            }
        }
    }
    let p = u.projected + art.projected; let w = u.whitened + art.projected;
    for (var r = 0u; r < counts[island] * width(); r += 1u) {
        let slot = island * u.contacts + r / width();
        let index = indices[slot]; let direction = directions[slot*width() + r%width()].xyz;
        for (var k = 0u; k < n; k += 1u) {
            var projected = 0.0;
            if (index >= u.rows) {
                if (r%width() == 0u) { projected = scalar_jacobians[(index-u.rows)*u.dofs+art.first+k]; }
            } else {
                let row = rows[index];
                projected = column(row.endpoints.w, art_index, row.point.xyz, direction, k)
                    - column(row.endpoints.z, art_index, row.point.xyz, direction, k);
            }
            if (!finite(projected)) { atomicOr(&status, NONFINITE); return; }
            scratch[p+r*n+k] = projected;
            var value = 0.0;
            if (root_scale > 0.0) { value = projected / root_scale; }
            for (var j = 0u; j < k; j += 1u) { value -= scratch[f+k*n+j] * scratch[w+r*n+j]; }
            let diagonal = scratch[f+k*n+k];
            if (diagonal == 0.0) { value = 0.0; } else { value /= diagonal; }
            if (!finite(value)) { atomicOr(&status, NONFINITE); return; }
            scratch[w+r*n+k] = value;
        }
    }
}

@compute @workgroup_size(64)
fn assemble(@builtin(global_invocation_id) id: vec3<u32>) {
    let island = id.x;
    if (atomicLoad(&status) != 0u || island >= island_count[0]) { return; }
    let stride = width() * u.contacts; let nrows = counts[island] * width();
    let begin = art_offsets[island]; let end = art_offsets[island+1u];
    if (begin > end || end > u.arts) { atomicOr(&status, INPUT); return; }
    for (var r = 0u; r < nrows; r += 1u) {
        let slot = island * u.contacts + r/width();
        var velocity = free[island*stride+r];
        if u.impact == 0u && block_kinds[slot] == 0u { velocity = data(slot,2u+r%2u); }
        if block_kinds[slot] == 0u {
            let row = rows[indices[slot]];
            velocity += dot(directions[width()*slot + r%width()].xyz,
                boundary_change(row.endpoints.w, row.point.xyz) - boundary_change(row.endpoints.z, row.point.xyz));
        }
        if u.impact != 3u {
        for (var member = begin; member < end; member += 1u) {
            let art = arts[art_indices[member]];
            for (var k = 0u; k < art.width; k += 1u) {
                var change = predicted[art.first+k] - input_rates[art.first+k];
                if (u.retained_delta & 1u) != 0u { change = predictor_delta[art.first+k]; }
                velocity += scratch[u.projected + art.projected + r*art.width+k] * change;
            }
        }
        }
        if u.impact == 2u && block_kinds[slot] == 0u && r%3u == 0u {
            velocity = bitcast<f32>(predicted_contacts[indices[slot]].y);
        }
        if (!finite(velocity)) { atomicOr(&status, NONFINITE); return; }
        free[island*stride+r] = velocity;
        if u.impact != 0u && block_kinds[slot] == 0u {
            if r%3u == 0u { set_data(slot,3u,velocity); }
            if r%3u == 2u {
                disk_input[slot*8u+5u] = u32(length(vec2<f32>(free[island*stride+r-1u],velocity)) > 1e-4);
            }
        }
        for (var c = 0u; c <= r; c += 1u) {
            var value = 0.0;
            for (var member = begin; member < end; member += 1u) {
                let art = arts[art_indices[member]]; let w = u.whitened + art.projected;
                for (var k = 0u; k < art.width; k += 1u) { value += scratch[w+r*art.width+k] * scratch[w+c*art.width+k]; }
            }
            if (!finite(value)) { atomicOr(&status, NONFINITE); return; }
            gram[island*stride*stride+r*stride+c] = value;
            gram[island*stride*stride+c*stride+r] = value;
        }
    }
}

@compute @workgroup_size(64)
fn apply(@builtin(global_invocation_id) id: vec3<u32>) {
    let art_index = id.x;
    if (atomicLoad(&status) != 0u || art_index >= u.arts) { return; }
    let art = arts[art_index]; let n = art.width; let island = art_islands[art_index];
    for (var step = 0u; step < n; step += 1u) {
        let k = n - step - 1u; let index=art.first+k; var value = vec2<f32>(0.0);
        if (island != NONE && scratch[u.scales + art_index] > 0.0) {
            for (var r = 0u; r < counts[island] * width(); r += 1u) {
                let slot=island*u.contacts+r/width();
                let weight=scratch[u.whitened+art.projected+r*n+k];
                if weight==0.0 { continue; }
                var multiplier=vec2<f32>(impulse(slot,r%width()),impulse_residue(slot,r%width()));
                if u.impact==0u && r%2u==0u && block_kinds[slot]!=0u {
                    multiplier.y=impulse(slot,1u);
                }
                value=response_add(index,value,response_product(index,weight,multiplier));
            }
            value=response_divide(index,value,scratch[u.scales+art_index]);
            for (var j = k+1u; j < n; j += 1u) {
                let entry=scratch[art.mass+j*n+k];
                if entry==0.0 { continue; }
                let previous=vec2<f32>(bitcast<f32>(atomicLoad(&delta[art.first+j])),corrected[art.first+j]);
                value=response_add(index,value,-response_product(index,entry,previous));
            }
            let diagonal = scratch[art.mass+k*n+k];
            if (diagonal == 0.0) { value = vec2<f32>(0.0); } else { value=response_divide(index,value,diagonal); }
        }
        if !finite(value.x) || !finite(value.y) { atomicOr(&status,NONFINITE); return; }
        atomicStore(&delta[index],bitcast<u32>(value.x)); corrected[index]=value.y;
    }
    for (var k=0u;k<n;k++) {
        let index=art.first+k;
        var rate=vec2<f32>(bitcast<f32>(atomicLoad(&delta[index])),corrected[index]);
        let change=rate.x;
        if u.impact != 3u {
            if (u.retained_delta & 1u) != 0u && ((u.retained_delta & 2u) == 0u || island != NONE) {
                rate=response_add(index,rate,vec2<f32>(predictor_delta[index],0.0));
                rate=response_add(index,rate,vec2<f32>(input_rates[index],0.0));
            } else {
                rate=response_add(index,rate,vec2<f32>(predicted[index],0.0));
            }
        }
        let published=response_round(index,rate.x+rate.y);
        if !finite(published) { atomicOr(&status,NONFINITE); return; }
        atomicStore(&delta[index],bitcast<u32>(change)); corrected[index]=published;
    }
}

@compute @workgroup_size(64)
fn readout(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = id.x; let island = slot / u.contacts; let c = slot % u.contacts;
    if (atomicLoad(&status) != 0u || island >= island_count[0]) { return; }
    if (c >= counts[island]) { return; }
    let index = indices[slot];
    if (index >= u.rows) { scalar_solutions[index-u.rows] = scalar_solution(slot); return; }
    var value = vec3<f32>(0.0);
    for (var i = 0u; i < width(); i += 1u) { value += directions[width()*slot+i].xyz*impulse(slot,i); }
    if (!finite3(value)) { atomicOr(&status, NONFINITE); return; }
    world_impulses[index] = vec4<f32>(value, 0.0);
}
