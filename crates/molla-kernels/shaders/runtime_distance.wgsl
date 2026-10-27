struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct Velocity { linear: vec4<f32>, angular: vec4<f32> }
struct Island { art_start: u32, art_count: u32, row_start: u32, row_count: u32, dofs: u32, scratch: u32, response_rank: u32, pad1: u32 }
struct Art { global: u32, width: u32, mass: u32, local: u32 }
struct Endpoint { body: i32, local: u32, width: u32, jacobian: u32, root_coordinate: i32, boundary: u32, pad1: u32, pad2: u32 }
struct Row { parent: Endpoint, child: Endpoint, a: vec4<f32>, b: vec4<f32>, kind: u32, coordinate: u32, dof: u32, bound: f32,
    primary: vec4<f32>, transverse0: vec4<f32>, transverse1: vec4<f32>, angles: vec2<f32>, enabled: vec2<u32>,
    a_low: vec4<f32>, b_low: vec4<f32>, primary_low: vec4<f32>, transverse0_low: vec4<f32> }
struct Params { dt: f32, weight: f32, substep: u32, count: u32, dofs: u32, retained_coordinates: u32, contact_prediction: u32, pad2: u32 }
struct Sample { force: vec4<f32>, tag: vec4<u32> }
struct ScalarRow { endpoints: vec4<u32>, velocity: vec4<f32> }
@group(0) @binding(0) var<storage, read> islands: array<Island>;
@group(0) @binding(1) var<storage, read> arts: array<Art>;
@group(0) @binding(2) var<storage, read> rows: array<Row>;
@group(0) @binding(3) var<storage, read> mass: array<f32>;
@group(0) @binding(4) var<storage, read> jacobians: array<f32>;
@group(0) @binding(5) var<storage, read> poses: array<Transform>;
@group(0) @binding(6) var<storage, read> coordinates: array<f32>;
@group(0) @binding(7) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(8) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(9) var<storage, read_write> samples: array<Sample>;
@group(0) @binding(10) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(11) var<uniform> params: Params;
@group(0) @binding(12) var<storage, read> active_islands: array<u32>;
@group(0) @binding(13) var<storage, read> body_velocities: array<Velocity>;
@group(0) @binding(14) var<storage, read> centers: array<vec4<f32>>;
@group(0) @binding(15) var<storage, read> input_velocities: array<f32>;
@group(0) @binding(16) var<storage, read_write> velocity_delta: array<f32>;
@group(0) @binding(17) var<storage, read_write> row_forces: array<atomic<u32>>;
@group(0) @binding(18) var<storage, read_write> scalar_rows: array<ScalarRow>;
@group(0) @binding(19) var<storage, read_write> scalar_jacobians: array<f32>;
@group(0) @binding(20) var<storage, read_write> scalar_directions: array<vec4<f32>>;
@group(0) @binding(21) var<storage, read> coordinate_residues: array<f32>;
@group(0) @binding(22) var<storage, read> boundary_acceleration: array<Velocity>;
@group(0) @binding(23) var<storage, read_write> scalar_positions: array<ScalarRow>;
@group(0) @binding(24) var<storage, read> position_residues: array<vec4<f32>>;
@group(0) @binding(25) var<storage, read> body_rotations: array<QuaternionPair>;
@group(0) @binding(26) var<storage, read> centers_low: array<vec4<f32>>;
@group(0) @binding(27) var<storage, read> prediction_displacement: array<f32>;

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn anchor(e: Endpoint, local: vec3<f32>, slot: u32) -> mat2x3<f32> {
    if e.body < 0 { return mat2x3<f32>(local,vec3<f32>(0.0)); }
    let body = u32(e.body);
    let pose = poses[body];
    var base = pose.position.xyz;
    var residue = vec3<f32>(0.0);
    var arm = rotate(pose.rotation,local);
    if params.retained_coordinates != 0u { residue = position_residues[body].xyz; }
    if e.root_coordinate >= 0 {
        let c = u32(e.root_coordinate);
        base = vec3<f32>(coordinates[c], coordinates[c+1u], coordinates[c+2u]);
        arm = rotate(pose.rotation,local-centers[body].xyz);
        if params.retained_coordinates != 0u {
            residue = vec3<f32>(coordinate_residues[c],coordinate_residues[c+1u],coordinate_residues[c+2u]);
        }
    }
    var high = vec3<f32>(0.0);
    for (var axis = 0u; axis < 3u; axis++) {
        let sum = radial_difference(slot,base[axis],-arm[axis]);
        high[axis] = sum.x;
        residue[axis] += sum.y;
    }
    return mat2x3<f32>(high,residue);
}
fn column(e: Endpoint, c: u32, direction: vec3<f32>, angular: bool) -> f32 {
    if c < e.local || c >= e.local + e.width { return 0.0; }
    let offset = e.jacobian + c - e.local + select(0u,3u*e.width,angular);
    return dot(direction, vec3<f32>(jacobians[offset], jacobians[offset+e.width], jacobians[offset+2u*e.width]));
}
fn anchor_velocity(e: Endpoint, point: vec3<f32>) -> vec3<f32> {
    if e.body < 0 { return vec3<f32>(0.0); }
    let body = u32(e.body);
    let offset = rotate(poses[body].rotation, point - centers[body].xyz);
    var velocity = body_velocities[body].linear.xyz + cross(body_velocities[body].angular.xyz, offset);
    if e.boundary != 0xffffffffu {
        let boundary = e.boundary;
        let arm = poses[body].position.xyz - poses[boundary].position.xyz
            + rotate(poses[body].rotation, point) - rotate(poses[boundary].rotation, centers[boundary].xyz);
        let acceleration = boundary_acceleration[boundary];
        velocity += (acceleration.linear.xyz + cross(acceleration.angular.xyz, arm)) * params.dt;
    }
    return velocity;
}
fn finite(x: f32) -> bool { return abs(x) <= 3.402823e38; }
fn radial_round(slot: u32, value: f32) -> f32 {
    atomicStore(&row_forces[slot],bitcast<u32>(value));
    return bitcast<f32>(atomicLoad(&row_forces[slot]));
}
fn radial_difference(slot: u32, a: f32, b: f32) -> vec2<f32> {
    let value = radial_round(slot,a-b);
    let virtual_b = radial_round(slot,a-value);
    let virtual_a = radial_round(slot,value+virtual_b);
    return vec2<f32>(value,(a-virtual_a)+(virtual_b-b));
}
fn rounded(slot: u32, value: f32) -> f32 { return radial_round(slot,value); }
fn two_sum(slot: u32, a: f32, b: f32) -> vec2<f32> { return radial_difference(slot,a,-b); }
fn weld_rotation(slot: u32, e: Endpoint, high: vec4<f32>, low: vec4<f32>) -> QuaternionPair {
    let local = QuaternionPair(high,low);
    if e.body < 0 { return local; }
    return multiply(slot,body_rotations[u32(e.body)],local);
}
fn weld_anchor(slot: u32, e: Endpoint, high: vec3<f32>, low: vec3<f32>) -> mat2x3<f32> {
    if e.body < 0 { return mat2x3<f32>(high,low); }
    let body = u32(e.body);
    var base = poses[body].position.xyz;
    var residue = position_residues[body].xyz;
    var local = QuaternionPair(vec4<f32>(high,0.0),vec4<f32>(low,0.0));
    if e.root_coordinate >= 0 {
        let coord = u32(e.root_coordinate);
        base = vec3<f32>(coordinates[coord],coordinates[coord+1u],coordinates[coord+2u]);
        residue = vec3<f32>(coordinate_residues[coord],coordinate_residues[coord+1u],coordinate_residues[coord+2u]);
        for (var lane = 0u; lane < 3u; lane++) {
            let difference = radial_difference(slot,high[lane],centers[body][lane]);
            let normalized = two_sum(slot,difference.x,difference.y+low[lane]-centers_low[body][lane]);
            local.high[lane] = normalized.x;
            local.low[lane] = normalized.y;
        }
    }
    let rotation = body_rotations[body];
    let inverse = QuaternionPair(vec4<f32>(-rotation.high.xyz,rotation.high.w),vec4<f32>(-rotation.low.xyz,rotation.low.w));
    let arm = multiply(slot,multiply(slot,rotation,local),inverse);
    var result = mat2x3<f32>(vec3<f32>(0.0),vec3<f32>(0.0));
    for (var lane = 0u; lane < 3u; lane++) {
        let sum = two_sum(slot,base[lane],arm.high[lane]);
        let normalized = two_sum(slot,sum.x,sum.y+residue[lane]+arm.low[lane]);
        result[0][lane] = normalized.x;
        result[1][lane] = normalized.y;
    }
    return result;
}
fn radial_gap(delta: vec3<f32>, residue: vec3<f32>, bound: f32, distance: f32, slot: u32) -> f32 {
    let magnitude = max(bound,distance);
    if magnitude == 0.0 { return 0.0; }
    let exponent = frexp(magnitude).exp;
    let radius = ldexp(bound,-exponent);
    let point = ldexp(delta,vec3<i32>(-exponent));
    let low = ldexp(residue,vec3<i32>(-exponent));
    let rr = radial_round(slot,radius*radius);
    let xx = radial_round(slot,point.x*point.x);
    let yy = radial_round(slot,point.y*point.y);
    let zz = radial_round(slot,point.z*point.z);
    let first = radial_difference(slot,rr,xx);
    let second = radial_difference(slot,first.x,yy);
    let third = radial_difference(slot,second.x,zz);
    let error = first.y+second.y+third.y + fma(radius,radius,-rr)
        - fma(point.x,point.x,-xx) - fma(point.y,point.y,-yy) - fma(point.z,point.z,-zz);
    let numerator = radial_difference(slot,third.x,dot(2.0*point+low,low));
    let denominator = radial_difference(slot,radius,-ldexp(distance,-exponent));
    let quotient = radial_round(slot,numerator.x/denominator.x);
    let remainder = fma(-quotient,denominator.x,numerator.x)
        + numerator.y+error-quotient*denominator.y;
    return ldexp(quotient+remainder/denominator.x,exponent);
}
fn bilateral(row: Row) -> bool {
    return (row.kind >= 3u && row.kind <= 6u) || row.kind == 10u || row.kind==23u || row.kind==24u
        || (row.kind >= 11u && row.kind <= 12u && row.enabled.x == 0u)
        || (drive_row(row.kind) && !drive_active(row))
        || ((speed_row(row.kind) || loop_speed_row(row.kind)) && row.enabled.x==0u);
}
fn minimum_norm_impulses(gram: u32, r: u32, impulse: u32, residuals: u32, search: u32, product: u32) {
    for (var i = 0u; i < r; i++) {
        var value = 0.0;
        for (var j = 0u; j < r; j++) { value += scratch[gram+i*r+j]*scratch[impulse+j]; }
        scratch[residuals+i] = value;
    }
    var norm = 0.0;
    for (var i = 0u; i < r; i++) {
        scratch[impulse+i] = 0.0;
        scratch[search+i] = scratch[residuals+i];
        norm += scratch[residuals+i]*scratch[residuals+i];
    }
    let tolerance = max(1e-12*norm,1e-20);
    for (var iteration = 0u; iteration < 256u && norm > tolerance; iteration++) {
        var denominator = 0.0;
        var curvature = 0.0;
        for (var i = 0u; i < r; i++) {
            var value = 0.0;
            for (var j = 0u; j < r; j++) { value += scratch[gram+i*r+j]*scratch[search+j]; }
            scratch[product+i] = value;
            denominator += scratch[search+i]*value;
            curvature += abs(scratch[gram+i*r+i])*scratch[search+i]*scratch[search+i];
        }
        if !(denominator > 1e-6*curvature) { break; }
        let alpha = norm/denominator;
        var next_norm = 0.0;
        for (var i = 0u; i < r; i++) {
            scratch[impulse+i] += alpha*scratch[search+i];
            scratch[residuals+i] -= alpha*scratch[product+i];
            next_norm += scratch[residuals+i]*scratch[residuals+i];
        }
        let beta = next_norm/norm;
        for (var i = 0u; i < r; i++) { scratch[search+i] = scratch[residuals+i]+beta*scratch[search+i]; }
        norm = next_norm;
    }
}

fn solved_bilateral(row: Row, flags: f32) -> bool {
    if row.kind==9u || row.kind==10u || euler_row(row.kind) { return flags<=1.0; }
    return bilateral(row);
}
fn drive_row(kind: u32) -> bool { return kind==13u || kind==14u; }
fn speed_row(kind: u32) -> bool { return kind==15u || kind==16u; }
fn loop_speed_row(kind: u32) -> bool { return kind>=19u && kind<=22u; }
fn linear_loop_row(kind: u32) -> bool { return kind==12u || kind==14u || kind==21u || kind==22u; }
fn euler_row(kind: u32) -> bool { return kind==17u || kind==18u || kind==23u || kind==24u; }
fn direct_row(kind: u32) -> bool { return kind==7u || kind==8u || speed_row(kind) || euler_row(kind); }
fn direct_sign(kind: u32) -> f32 { return select(1.0,-1.0,kind==8u || kind==16u || kind==18u); }
fn angular_row(kind: u32) -> bool { return (kind >= 9u && kind <= 11u) || kind == 13u || kind==19u || kind==20u; }
fn drive_active(row: Row) -> bool {
    return row.enabled.x == 4u || (row.enabled.x != 0u && any(row.angles > vec2<f32>(0.0)));
}
fn drive_diagonal(row: Row) -> f32 {
    if row.enabled.x == 4u || !drive_active(row) { return 0.0; }
    return 1.0/(params.dt*(row.angles.y+params.dt*row.angles.x));
}
fn acceleration_drive(row: Row) -> bool {
    return drive_row(row.kind) && row.transverse1.w==1.0 && row.enabled.x!=4u && drive_active(row);
}
fn drive_bounds(row: Row, mobility: f32) -> vec2<f32> {
    if !drive_active(row) || (acceleration_drive(row) && mobility==0.0) { return vec2<f32>(0.0); }
    let cap=row.transverse1.z;
    if row.enabled.x == 4u { return vec2<f32>(params.dt*clamp(row.bound+row.b.w,-cap,cap)); }
    return vec2<f32>(-params.dt*cap,params.dt*cap);
}
fn quaternion_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w*b.xyz+b.w*a.xyz+cross(a.xyz,b.xyz),a.w*b.w-dot(a.xyz,b.xyz));
}
fn frame_rotation(e: Endpoint, local: vec4<f32>) -> vec4<f32> {
    if e.body < 0 { return local; }
    return quaternion_mul(poses[u32(e.body)].rotation,local);
}
fn angular_velocity(e: Endpoint) -> vec3<f32> {
    if e.body < 0 { return vec3<f32>(0.0); }
    var velocity = body_velocities[u32(e.body)].angular.xyz;
    if e.boundary != 0xffffffffu { velocity += boundary_acceleration[e.boundary].angular.xyz * params.dt; }
    return velocity;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) id: vec3<u32>) {
    execute(id.x, false);
}

@compute @workgroup_size(64)
fn prepare_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    execute(id.x, true);
}

fn interval_prediction(row: Row, trial: u32, position_trial: u32, slot: u32) -> EulerBounds {
    let rate=vec3<f32>(scratch[trial+row.dof],scratch[trial+row.dof+1u],scratch[trial+row.dof+2u]);
    var initial=QuaternionPair(vec4<f32>(coordinates[row.coordinate],coordinates[row.coordinate+1u],
        coordinates[row.coordinate+2u],coordinates[row.coordinate+3u]),vec4<f32>(0.0));
    if params.retained_coordinates!=0u {
        initial.low=vec4<f32>(coordinate_residues[row.coordinate],coordinate_residues[row.coordinate+1u],
            coordinate_residues[row.coordinate+2u],coordinate_residues[row.coordinate+3u]);
    }
    let displacement=vec3<f32>(scratch[position_trial+row.dof],scratch[position_trial+row.dof+1u],
        scratch[position_trial+row.dof+2u]);
    if any(displacement!=vec3<f32>(0.0)) {
        initial=normalize_pair(slot,multiply(slot,QuaternionPair(vec4<f32>(0.5*displacement,1.0),
            vec4<f32>(0.0)),initial));
    }
    let half_rate=0.5*rate;
    var delta=QuaternionPair(vec4<f32>(half_rate*params.dt,1.0),vec4<f32>(0.0));
    delta.low=vec4<f32>(fma(half_rate,vec3<f32>(params.dt),-delta.high.xyz),0.0);
    let predicted=normalize_pair(slot,multiply(slot,delta,initial));
    return euler_bounds_pair(slot,predicted,row.primary+row.primary_low,
        row.transverse0.xy+row.transverse0_low.xy,false);
}

fn interval_feasible(row: Row, trial: u32, position_trial: u32, slot: u32) -> bool {
    let geometry=interval_prediction(row,trial,position_trial,slot);
    let pitch=row.transverse0.xy+row.transverse0_low.xy;
    let bounds=row.primary+row.primary_low;
    let lower=vec3<f32>(bounds.x,pitch.x,bounds.z);
    let upper=vec3<f32>(bounds.y,pitch.y,bounds.w);
    for (var axis=0u; axis<3u; axis++) {
        if geometry.scales.z==0.0 && axis!=1u { continue; }
        let angle=geometry.angles[axis];
        if !finite(angle) { return false; }
        let scale=max(geometry.scales[axis],1e-6);
        if finite(lower[axis]) && (lower[axis]-angle)*scale>1e-7*(1.0+abs(lower[axis])) { return false; }
        if finite(upper[axis]) && (angle-upper[axis])*scale>1e-7*(1.0+abs(upper[axis])) { return false; }
    }
    if geometry.scales.z==0.0 {
        let phase=geometry.phase;
        if !finite(phase.x) { return false; }
        if finite(phase.y) && phase.y-phase.x>1e-7*(1.0+abs(phase.y)) { return false; }
        if finite(phase.z) && phase.x-phase.z>1e-7*(1.0+abs(phase.z)) { return false; }
    }
    return true;
}

fn load_prediction(island: Island, trial: u32, position_trial: u32, contact: bool) {
    for (var a=0u;a<island.art_count;a++) {
        let art=arts[island.art_start+a];
        for (var c=0u;c<art.width;c++) {
            scratch[trial+art.local+c]=velocities[art.global+c];
            var displacement=0.0;
            if contact { displacement=prediction_displacement[art.global+c]; }
            scratch[position_trial+art.local+c]=displacement;
        }
    }
}

@compute @workgroup_size(64)
fn verify_prediction(@builtin(global_invocation_id) id: vec3<u32>) {
    let index=id.x;
    if index>=params.count || atomicLoad(&status)!=0u || active_islands[index]==0u { return; }
    let island=islands[index]; let n=island.dofs; let r=island.row_count;
    let trial=island.scratch+n*n+2u*r*n+r*r+9u*r;
    let position_trial=trial+n;
    load_prediction(island,trial,position_trial,true);
    for (var i=0u;i<r;i++) {
        let row=rows[island.row_start+i];
        if euler_row(row.kind) && ((row.enabled.y>>1u)&7u)==0u
            && !interval_feasible(row,trial,position_trial,4u*(island.row_start+i)+3u) {
            atomicOr(&status,512u); return;
        }
    }
}

fn execute(index: u32, export_rows: bool) {
    if index >= params.count || atomicLoad(&status) != 0u { return; }
    if active_islands[index] == 0u { return; }
    let island = islands[index];
    let n = island.dofs;
    let r = island.row_count;
    let factor = island.scratch;
    let projected = factor + n*n;
    let response = projected + r*n;
    let gram = response + r*n;
    let rhs = gram + r*r;
    let impulse = rhs + r;
    let directions = impulse + r;
    let residuals = directions + 3u*r;
    let search = residuals + r;
    let product = search + r;
    let row_flags = product + r;
    let trial = row_flags + r;
    let position_trial=trial+n;
    let singular_root=island.art_count==1u && n==6u && island.response_rank<6u;
    var nonlinear=false;
    if !export_rows || params.contact_prediction!=0u {
        for (var i=0u;i<r;i++) {
            let row=rows[island.row_start+i];
            nonlinear=nonlinear || (euler_row(row.kind) && ((row.enabled.y>>1u)&7u)==0u);
        }
    }
    load_prediction(island,trial,position_trial,export_rows && params.contact_prediction!=0u);
    var acceleration=false;
    for(var i=0u;i<r;i++) {acceleration=acceleration || acceleration_drive(rows[island.row_start+i]);}
    var floor = 0.0;
    if !export_rows || acceleration {
    for (var i = 0u; i < n*n; i++) { scratch[factor+i] = 0.0; }
    for (var a = 0u; a < island.art_count; a++) {
        let art = arts[island.art_start+a];
        for (var i = 0u; i < art.width; i++) {
            for (var j = 0u; j < art.width; j++) {
                scratch[factor+(art.local+i)*n+art.local+j] = mass[art.mass+i*art.width+j];
            }
        }
    }
    if singular_root {
        var matrix: array<f32,36>;
        for (var i=0u;i<36u;i++) { matrix[i]=scratch[factor+i]; }
        let inverse=joint_psd_inverse_rank(matrix,6u,island.response_rank);
        if !inverse.valid { atomicOr(&status,256u); return; }
        for (var i=0u;i<36u;i++) { scratch[factor+i]=inverse.matrix[i]; }
    } else {
    var scale = 0.0;
    for (var i = 0u; i < n; i++) { scale = max(scale, abs(scratch[factor+i*n+i])); }
    floor = scale * 1e-10;
    for (var i = 0u; i < n; i++) {
        for (var j = 0u; j <= i; j++) {
            var value = scratch[factor+i*n+j];
            for (var k = 0u; k < j; k++) { value -= scratch[factor+i*n+k]*scratch[factor+j*n+k]; }
            if !finite(value) || (i == j && value < -scale * 1e-5) { atomicOr(&status, 256u); return; }
            if i == j { scratch[factor+i*n+j] = sqrt(max(value, 0.0)); }
            else {
                let diagonal = scratch[factor+j*n+j];
                var entry = 0.0;
                if diagonal*diagonal > floor { entry = value / diagonal; }
                scratch[factor+i*n+j] = entry;
            }
        }
    }
    }
    }
    var prediction_converged=!nonlinear;
    for (var prediction_iteration=0u;prediction_iteration<select(1u,16u,nonlinear && !export_rows);prediction_iteration++) {
    for (var i = 0u; i < r; i++) {
        let row_id = island.row_start+i;
        let row = rows[row_id];
        scratch[row_flags+i]=select(2.0,1.0,bilateral(row));
        scratch[search+i]=1.0;
        if params.substep == 0u { samples[row_id] = Sample(vec4<f32>(0.0), vec4<u32>(0u)); }
        var direction = vec3<f32>(0.0);
        var desired = 0.0;
        var position_bias = 0.0;
        var position_low = 0.0;
        var parent_couple = vec3<f32>(0.0);
        var direct_gradient=vec3<f32>(0.0);
        if drive_row(row.kind) && !drive_active(row) {
            for (var c=0u;c<n;c++) { scratch[projected+i*n+c]=0.0;scratch[response+i*n+c]=0.0; }
            for (var c=0u;c<3u;c++) { scratch[directions+3u*i+c]=0.0; }
            scratch[rhs+i]=0.0;scratch[impulse+i]=0.0;
            continue;
        }
        if row.kind <= 6u {
            let slot = 4u*row_id+3u;
            var child = anchor(row.child,row.b.xyz,slot);
            var parent = anchor(row.parent,row.a.xyz,slot);
            if params.retained_coordinates!=0u {
                child=weld_anchor(slot,row.child,row.b.xyz,row.b_low.xyz);
                parent=weld_anchor(slot,row.parent,row.a.xyz,row.a_low.xyz);
            }
            var delta = vec3<f32>(0.0);
            var residue = child[1]-parent[1];
            for (var axis = 0u; axis < 3u; axis++) {
                let difference = radial_difference(slot,child[0][axis],parent[0][axis]);
                residue[axis] += difference.y;
                let normalized = radial_difference(slot,difference.x,-residue[axis]);
                delta[axis] = normalized.x;
                residue[axis] = normalized.y;
            }
            let length = length(delta);
            direction = vec3<f32>(row.a.w, 0.0, 0.0);
            if length > 1e-12 { direction = delta / length; }
            let gap = radial_gap(delta,residue,row.bound,length,slot)+row.b.w;
            desired = gap/params.dt;
            position_bias = -gap;
            if row.kind == 2u { direction = -direction; desired = -desired; position_bias = -position_bias; }
            if row.kind >= 4u {
                direction = vec3<f32>(0.0);
                direction[row.kind-4u] = 1.0;
                position_bias = delta[row.kind-4u]+residue[row.kind-4u];
                desired = -position_bias/params.dt;
            }
        } else if row.kind <= 8u {
            let sign = select(1.0, -1.0, row.kind == 8u);
            var gap = row.bound-coordinates[row.coordinate];
            if params.retained_coordinates != 0u { gap -= coordinate_residues[row.coordinate]; }
            gap += row.b.w;
            desired = sign * gap/params.dt;
            position_bias = -sign * gap;
        } else if speed_row(row.kind) {
            desired=row.bound+row.b.w;
        } else if euler_row(row.kind) {
            if row.enabled.x>=3u { atomicOr(&status,512u); return; }
            let point_mask=(row.enabled.y>>1u)&7u;
            let pitch_point=(point_mask&2u)!=0u;
            var geometry=euler_bounds(4u*row_id+3u,row.coordinate,row.primary+row.primary_low,
                row.transverse0.xy+row.transverse0_low.xy,pitch_point);
            if nonlinear && point_mask==0u { geometry=interval_prediction(row,trial,position_trial,4u*row_id+3u); }
            let axis=row.enabled.x;
            let sign=direct_sign(row.kind);
            let singular=geometry.scales.z==0.0;
            let phase_row=(row.enabled.y&1u)!=0u;
            let positive_phase=geometry.phase.w>0.0;
            let phase_active=(row.enabled.y&select(128u,64u,positive_phase))!=0u;
            let pitch_bounds=row.transverse0.xy+row.transverse0_low.xy;
            let bounds=row.primary+row.primary_low;
            let lower=vec3<f32>(bounds.x,pitch_bounds.x,bounds.z);
            let upper=vec3<f32>(bounds.y,pitch_bounds.y,bounds.w);
            let points=countOneBits(point_mask); var free_axis=0u;
            for (var a=0u; a<3u; a++) {
                if (point_mask&(1u<<a))==0u { free_axis=a; }
            }
            let pair_row=points==2u && (row.kind==23u || row.kind==24u);
            let pair_bound=points==2u && !phase_row && axis==free_axis && (row.kind==17u || row.kind==18u);
            let pitch_pole=pitch_point && abs(cos(pitch_bounds.x))<=1.9073486328125e-6;
            let plane_row=row.kind==23u && axis!=1u && singular && !pitch_pole;
            if (phase_row && (!singular || !phase_active || points==2u))
                || (!phase_row && singular && axis!=1u && !plane_row && !pair_row && !pair_bound) {
                for (var c=0u;c<n;c++) { scratch[projected+i*n+c]=0.0;scratch[response+i*n+c]=0.0; }
                for (var c=0u;c<3u;c++) { scratch[directions+3u*i+c]=0.0; }
                scratch[rhs+i]=0.0;scratch[impulse+i]=0.0;scratch[row_flags+i]=0.0;
                continue;
            }
            var gap=sign*((row.bound-geometry.angles[axis])+row.b.w);
            direct_gradient=sign*geometry.gradients[axis];
            var scale=geometry.scales[axis];
            if (!singular || axis==1u) && !phase_row && !pair_row && !pair_bound {
                gap=sign*euler_reference_gap(4u*row_id+3u,(row.bound-geometry.angles[axis])+row.b.w,
                    vec2<f32>(geometry.sine[axis],geometry.cosine[axis]),
                    vec2<f32>(geometry.sine_low[axis],geometry.cosine_low[axis]),row.angles,row.transverse0.zw,1.0);
            }
            if plane_row {
                let plane=euler_point_plane(4u*row_id+3u,row.coordinate,axis,
                    QuaternionPair(row.a,row.a_low),QuaternionPair(row.transverse1,row.b_low));
                direct_gradient=plane.xyz; gap=-plane.w; scale=1.0;
            }
            if phase_row {
                let equality=(row.enabled.y&select(32u,16u,positive_phase))!=0u;
                if equality && row.kind==18u {
                    for (var c=0u;c<n;c++) { scratch[projected+i*n+c]=0.0;scratch[response+i*n+c]=0.0; }
                    for (var c=0u;c<3u;c++) { scratch[directions+3u*i+c]=0.0; }
                    scratch[rhs+i]=0.0;scratch[impulse+i]=0.0;scratch[row_flags+i]=0.0;
                    continue;
                }
                scratch[row_flags+i]=select(2.0,1.0,equality);
                let pole=select(1u,0u,positive_phase);
                let reference=select(row.b.xy,row.a.zw,positive_phase);
                let reference_low=select(row.b_low.xy,row.a_low.zw,positive_phase);
                gap=sign*euler_reference_gap(4u*row_id+3u,(row.a[pole]-geometry.phase.x)+row.a_low[pole],
                    vec2<f32>(geometry.sine.x,geometry.cosine.x),
                    vec2<f32>(geometry.sine_low.x,geometry.cosine_low.x),reference,reference_low,1.0);
                direct_gradient=vec3<f32>(sign,0.0,0.0);
                scale=1.0;
            }
            if pair_bound {
                let coordinate=euler_pair_bound(4u*row_id+3u,row.coordinate,lower,upper,free_axis,
                    QuaternionPair(row.a,row.a_low),QuaternionPair(row.transverse1,row.b_low),
                    vec2<f32>(row.bound,row.b.w),row.angles,row.transverse0.zw);
                direct_gradient=coordinate.xyz*sign; gap=sign*coordinate.w; scale=1.0;
            } else if pair_row {
                let alignment=euler_pair_alignment(4u*row_id+3u,row.coordinate,
                    QuaternionPair(row.a,row.a_low),QuaternionPair(row.transverse1,row.b_low),free_axis);
                if alignment.info.z!=0u { atomicOr(&status,1024u); return; }
                let index=select(0u,1u,row.kind==24u);
                direct_gradient=alignment.rows[index].xyz; gap=-alignment.rows[index].w; scale=1.0;
            } else if axis==1u && (row.kind==23u || row.kind==24u) && abs(cos(row.bound+row.b.w))<=1.9073486328125e-6 {
                let alignment=euler_pitch_alignment(4u*row_id+3u,row.coordinate,
                    QuaternionPair(row.a,row.a_low),QuaternionPair(row.transverse1,row.b_low));
                if alignment.info.z!=0u { atomicOr(&status,1024u); return; }
                let index=select(0u,1u,row.kind==24u);
                direct_gradient=alignment.rows[index].xyz;
                gap=-alignment.rows[index].w;
                scale=1.0;
            }
            position_bias=-gap*scale;
            desired=-position_bias/params.dt;
            if nonlinear && point_mask==0u {
                let rate=vec3<f32>(scratch[trial+row.dof],scratch[trial+row.dof+1u],scratch[trial+row.dof+2u]);
                let half_delta=rate*(0.5*params.dt);
                direct_gradient=(direct_gradient-cross(half_delta,direct_gradient))/(1.0+dot(half_delta,half_delta));
                desired+=dot(direct_gradient,rate);
            }
        } else if row.kind <= 10u {
            var parent=frame_rotation(row.parent,row.a);
            var relative=quaternion_mul(vec4<f32>(-parent.xyz,parent.w),frame_rotation(row.child,row.b));
            if params.retained_coordinates!=0u {
                let slot=4u*row_id+3u;
                let parent_pair=normalize_pair(slot,weld_rotation(slot,row.parent,row.a,row.a_low));
                parent=normalize(parent_pair.high+parent_pair.low);
                let child=weld_rotation(slot,row.child,row.b,row.b_low);
                let inverse=QuaternionPair(vec4<f32>(-parent_pair.high.xyz,parent_pair.high.w),vec4<f32>(-parent_pair.low.xyz,parent_pair.low.w));
                let relative_pair=normalize_pair(slot,multiply(slot,inverse,child));
                relative=relative_pair.high+relative_pair.low;
            }
            if row.dof != 0u {
                let c = row.dof-1u;
                relative = vec4<f32>(coordinates[c],coordinates[c+1u],coordinates[c+2u],coordinates[c+3u]);
                if params.retained_coordinates!=0u {
                    relative+=vec4<f32>(coordinate_residues[c],coordinate_residues[c+1u],coordinate_residues[c+2u],coordinate_residues[c+3u]);
                }
            }
            let canonical = vec4<f32>(dot(relative.xyz,row.primary.xyz),dot(relative.xyz,row.transverse0.xyz),dot(relative.xyz,row.transverse1.xyz),relative.w);
            let geometry = cone_geometry(ConeRequest(canonical,row.primary,row.transverse0,row.transverse1,row.angles,row.enabled));
            if geometry.info.z != 0u { atomicOr(&status,1024u); return; }
            if row.coordinate >= geometry.info.x {
                for (var c=0u;c<n;c++) { scratch[projected+i*n+c]=0.0;scratch[response+i*n+c]=0.0; }
                for (var c=0u;c<3u;c++) { scratch[directions+3u*i+c]=0.0; }
                scratch[rhs+i]=0.0;scratch[impulse+i]=0.0;scratch[row_flags+i]=0.0;
                continue;
            }
            scratch[row_flags+i]=select(2.0,1.0,(geometry.info.y & (1u<<row.coordinate))!=0u);
            let cone = geometry.rows[row.coordinate];
            var magnitude = length(cone.xyz);
            if magnitude == 0.0 { magnitude = 1.0; }
            direction = -rotate(parent,cone.xyz/magnitude);
            desired = cone.w/(magnitude*params.dt);
            position_bias = -cone.w/magnitude;
        } else {
            let slot = 4u*row_id+3u;
            let parent_pair = normalize_pair(slot,weld_rotation(slot,row.parent,row.primary,row.primary_low));
            let parent = normalize(parent_pair.high+parent_pair.low);
            var axis = vec3<f32>(0.0);
            axis[row.coordinate] = 1.0;
            if angular_row(row.kind) {
                let child = weld_rotation(slot,row.child,row.transverse0,row.transverse0_low);
                let inverse = QuaternionPair(vec4<f32>(-parent_pair.high.xyz,parent_pair.high.w),vec4<f32>(-parent_pair.low.xyz,parent_pair.low.w));
                let relative_pair = normalize_pair(slot,multiply(slot,inverse,child));
                var relative = relative_pair.high+relative_pair.low;
                let canonical_sign=select(1.0,-1.0,relative.w<0.0);
                relative*=canonical_sign;
                let sine = length(relative.xyz);
                var rotation = 2.0*relative.xyz;
                var rotation_low=vec3<f32>(0.0);
                if sine<=0.125 {
                    let square=dot(relative.xyz,relative.xyz);
                    let tail=square*(1.0/3.0+square*(3.0/20.0+square*(5.0/56.0+square*35.0/576.0)));
                    for(var c=0u;c<3u;c++) {
                        let high=canonical_sign*relative_pair.high[c];
                        let low=canonical_sign*relative_pair.low[c];
                        let pair=two_sum(slot,2.0*high,2.0*low+(high+low)*tail);
                        rotation[c]=pair.x;rotation_low[c]=pair.y;
                    }
                } else { rotation = relative.xyz * (2.0*atan2(sine,relative.w)/sine); }
                let angle = length(rotation);
                let squared=angle*angle;
                var coefficient=1.0/12.0+squared*(1.0/720.0+squared*(1.0/30240.0+squared/1209600.0));
                if angle > 0.25 { coefficient = (1.0-0.5*angle/tan(0.5*angle))/(angle*angle); }
                direction = rotate(parent,axis + 0.5*cross(rotation,axis) + coefficient*cross(rotation,cross(rotation,axis)));
                position_bias = rotation[row.coordinate];
                position_low = rotation_low[row.coordinate];
            } else {
                let child = weld_anchor(slot,row.child,row.b.xyz,row.b_low.xyz);
                let a = weld_anchor(slot,row.parent,row.a.xyz,row.a_low.xyz);
                var delta_pair = QuaternionPair(vec4<f32>(0.0),vec4<f32>(0.0));
                for (var c = 0u; c < 3u; c++) {
                    let difference = radial_difference(slot,child[0][c],a[0][c]);
                    let residue = rounded(slot,child[1][c]-a[1][c]+difference.y);
                    let normalized = two_sum(slot,difference.x,residue);
                    delta_pair.high[c] = normalized.x;
                    delta_pair.low[c] = normalized.y;
                }
                let inverse = QuaternionPair(vec4<f32>(-parent_pair.high.xyz,parent_pair.high.w),vec4<f32>(-parent_pair.low.xyz,parent_pair.low.w));
                let local = multiply(slot,multiply(slot,inverse,delta_pair),parent_pair);
                let delta = delta_pair.high.xyz+delta_pair.low.xyz;
                direction = rotate(parent,axis);
                position_bias = local.high[row.coordinate];
                position_low = local.low[row.coordinate];
                parent_couple = cross(delta,direction);
            }
            if loop_speed_row(row.kind) {
                let sign=select(1.0,-1.0,row.kind==20u || row.kind==22u);
                direction*=sign;
                parent_couple*=sign;
                position_bias=0.0;
                desired=row.bound+row.b.w;
            } else {
                let difference = radial_difference(slot,position_bias,row.bound);
                let residue = rounded(slot,difference.y+position_low-row.b.w);
                let sign = select(1.0,-1.0,row.kind <= 12u && row.enabled.x == 2u);
                position_bias = sign*rounded(slot,difference.x+residue);
                direction *= sign;
                parent_couple *= sign;
                desired = -position_bias/params.dt;
            }
            if drive_row(row.kind) {
                if row.enabled.x == 4u { desired=0.0; }
                else {
                    let target_rate=select(0.0,row.transverse1.y,row.enabled.x == 2u || row.enabled.x == 3u);
                    desired=(row.angles.y*target_rate-row.angles.x*position_bias)/(row.angles.y+params.dt*row.angles.x);
                }
                if !finite(drive_diagonal(row)) || !finite(desired) { atomicOr(&status,512u);return; }
            }
        }
        for (var c = 0u; c < 3u; c++) { scratch[directions+3u*i+c] = direction[c]; }
        for (var c = 0u; c < n; c++) {
            var value = column(row.child, c, direction,angular_row(row.kind))-column(row.parent, c, direction,angular_row(row.kind));
            if linear_loop_row(row.kind) { value -= column(row.parent,c,parent_couple,true); }
            if direct_row(row.kind) && !euler_row(row.kind) { value = select(0.0,direct_sign(row.kind),c == row.dof); }
            if euler_row(row.kind) {
                value=0.0;
                if c>=row.dof && c<row.dof+3u { value=direct_gradient[c-row.dof]; }
            }
            scratch[projected+i*n+c] = value;
            scratch[response+i*n+c] = value;
        }
        var rate = 0.0;
        if row.kind <= 6u || linear_loop_row(row.kind) {
            rate = dot(direction, anchor_velocity(row.child,row.b.xyz)-anchor_velocity(row.parent,row.a.xyz))
                - dot(parent_couple,angular_velocity(row.parent));
        }
        if angular_row(row.kind) { rate = dot(direction,angular_velocity(row.child)-angular_velocity(row.parent)); }
        for (var a = 0u; a < island.art_count; a++) {
            let art = arts[island.art_start+a];
            for (var c = 0u; c < art.width; c++) {
                let jacobian = scratch[projected+i*n+art.local+c];
                if !export_rows { rate += jacobian*velocity_delta[art.global+c]; }
                if direct_row(row.kind) { rate += jacobian*input_velocities[art.global+c]; }
                if export_rows {
                    if !finite(jacobian) { atomicOr(&status,512u); return; }
                    scalar_jacobians[row_id*params.dofs+art.global+c] = jacobian;
                }
            }
        }
        if export_rows {
            let free = rate-desired;
            if !finite(free) || !finite(position_bias) || !finite(direction.x) || !finite(direction.y) || !finite(direction.z) { atomicOr(&status,512u); return; }
            scalar_rows[row_id] = ScalarRow(vec4<u32>(bitcast<u32>(row.parent.body),bitcast<u32>(row.child.body),
                select(2u,1u,solved_bilateral(row,scratch[row_flags+i])),1u),vec4<f32>(free,0.0,0.0,0.0));
            scalar_positions[row_id] = ScalarRow(scalar_rows[row_id].endpoints,vec4<f32>(position_bias,0.0,0.0,0.0));
            if speed_row(row.kind) || loop_speed_row(row.kind)
                || (nonlinear && euler_row(row.kind) && ((row.enabled.y>>1u)&7u)==0u) {
                scalar_positions[row_id]=ScalarRow(vec4<u32>(0u),vec4<f32>(0.0));
            }
            if drive_row(row.kind) {
                let bounds=drive_bounds(row,1.0);
                scalar_rows[row_id].endpoints.z=3u;
                scalar_rows[row_id].velocity=vec4<f32>(free,drive_diagonal(row),bounds);
                scalar_positions[row_id]=ScalarRow(vec4<u32>(0u),vec4<f32>(0.0));
            }
            scalar_directions[row_id] = vec4<f32>(direction,f32(row.kind));
            if !acceleration {continue;}
        }
        scratch[rhs+i] = desired-rate;
        scratch[impulse+i] = 0.0;
        if singular_root {
            for (var c=0u;c<n;c++) {
                var value=0.0;
                for (var k=0u;k<n;k++) { value=fma(scratch[factor+c*n+k],scratch[projected+i*n+k],value); }
                scratch[response+i*n+c]=value;
            }
        } else {
        for (var c = 0u; c < n; c++) {
            var value = scratch[response+i*n+c];
            for (var k = 0u; k < c; k++) { value -= scratch[factor+c*n+k]*scratch[response+i*n+k]; }
            let diagonal = scratch[factor+c*n+c];
            scratch[response+i*n+c] = 0.0;
            if diagonal*diagonal > floor { scratch[response+i*n+c] = value/diagonal; }
        }
        for (var k = n; k > 0u; k--) {
            let c = k-1u;
            var value = scratch[response+i*n+c];
            for (var j = c+1u; j < n; j++) { value -= scratch[factor+j*n+c]*scratch[response+i*n+j]; }
            let diagonal = scratch[factor+c*n+c];
            scratch[response+i*n+c] = 0.0;
            if diagonal*diagonal > floor { scratch[response+i*n+c] = value/diagonal; }
        }
        }
    }
    if export_rows && !acceleration { return; }
    for (var i = 0u; i < r; i++) {
        for (var j = 0u; j < r; j++) {
            var value = 0.0;
            for (var c = 0u; c < n; c++) { value += scratch[projected+i*n+c]*scratch[response+j*n+c]; }
            scratch[gram+i*r+j] = value;
        }
    }
    if acceleration {
        for(var i=0u;i<r;i++) {scratch[product+i]=scratch[gram+i*r+i];scratch[residuals+i]=0.0;}
        for(var step=0u;step<r;step++) {
            var pivot=r;
            var best=1e-6;
            for(var i=0u;i<r;i++) {
                let row=rows[island.row_start+i];
                if !drive_row(row.kind) && solved_bilateral(row,scratch[row_flags+i]) && scratch[residuals+i]==0.0 && scratch[product+i]>0.0 {
                    let score=scratch[gram+i*r+i]/scratch[product+i];
                    if score>best {best=score;pivot=i;}
                }
            }
            if pivot==r {break;}
            let denominator=scratch[gram+pivot*r+pivot];
            for(var i=0u;i<r;i++) {
                if i==pivot {continue;}
                for(var j=0u;j<r;j++) {
                    if j!=pivot {scratch[gram+i*r+j]-=scratch[gram+i*r+pivot]*(scratch[gram+pivot*r+j]/denominator);}
                }
            }
            for(var i=0u;i<r;i++) {scratch[gram+i*r+pivot]=0.0;scratch[gram+pivot*r+i]=0.0;}
            scratch[residuals+pivot]=1.0;
        }
        for(var i=0u;i<r;i++) {
            let row=rows[island.row_start+i];
            if acceleration_drive(row) {
                let value=scratch[gram+i*r+i];
                if !finite(value) || value< -1e-5*scratch[product+i] {atomicOr(&status,512u);return;}
                scratch[search+i]=select(0.0,value,value>1e-6*scratch[product+i]);
            }
        }
    }
    if export_rows {
        for(var i=0u;i<r;i++) {
            let row=rows[island.row_start+i];
            if drive_row(row.kind) && drive_active(row) {
                scalar_rows[island.row_start+i].velocity.y=drive_diagonal(row)*scratch[search+i];
                let bounds=drive_bounds(row,scratch[search+i]);
                scalar_rows[island.row_start+i].velocity.z=bounds.x;
                scalar_rows[island.row_start+i].velocity.w=bounds.y;
            }
        }
        return;
    }
    for(var i=0u;i<r;i++) {
        if acceleration {
            for(var j=0u;j<r;j++) {
                var value=0.0;
                for(var c=0u;c<n;c++) {value+=scratch[projected+i*n+c]*scratch[response+j*n+c];}
                scratch[gram+i*r+j]=value;
            }
        }
        let row=rows[island.row_start+i];
        if drive_row(row.kind) {scratch[gram+i*r+i]+=drive_diagonal(row)*scratch[search+i];}
    }
    var bilateral_only = true;
    for (var i = 0u; i < r; i++) {
        bilateral_only = bilateral_only && solved_bilateral(rows[island.row_start+i],scratch[row_flags+i]);
    }
    if bilateral_only {
        var norm = 0.0;
        for (var i = 0u; i < r; i++) {
            scratch[residuals+i] = scratch[rhs+i];
            scratch[search+i] = scratch[rhs+i];
            norm += scratch[rhs+i]*scratch[rhs+i];
        }
        let tolerance = max(1e-12 * norm,1e-20);
        for (var iteration = 0u; iteration < 256u && norm > tolerance; iteration++) {
            var denominator = 0.0;
            for (var i = 0u; i < r; i++) {
                var value = 0.0;
                for (var j = 0u; j < r; j++) { value += scratch[gram+i*r+j]*scratch[search+j]; }
                scratch[product+i] = value;
                denominator += scratch[search+i]*value;
            }
            var curvature = 0.0;
            for (var i=0u;i<r;i++) {
                curvature += abs(scratch[gram+i*r+i])*scratch[search+i]*scratch[search+i];
            }
            if !(denominator > 1e-6*curvature) { break; }
            let alpha = norm/denominator;
            var next_norm = 0.0;
            for (var i = 0u; i < r; i++) {
                scratch[impulse+i] += alpha*scratch[search+i];
                scratch[residuals+i] -= alpha*scratch[product+i];
                next_norm += scratch[residuals+i]*scratch[residuals+i];
            }
            let beta = next_norm/norm;
            for (var i = 0u; i < r; i++) { scratch[search+i] = scratch[residuals+i]+beta*scratch[search+i]; }
            norm = next_norm;
        }
    } else {
    for (var iteration = 0u; iteration < 256u; iteration++) {
        for (var i = 0u; i < r; i++) {
            let diagonal = scratch[gram+i*r+i];
            let row=rows[island.row_start+i];
            var residual = scratch[rhs+i];
            for (var j = 0u; j < r; j++) { residual -= scratch[gram+i*r+j]*scratch[impulse+j]; }
            var value=0.0;
            if diagonal>0.0 { value = scratch[impulse+i]+residual/diagonal; }
            if drive_row(row.kind) {
                let bounds=drive_bounds(row,scratch[search+i]);
                if diagonal<=0.0 { value=select(select(0.0,bounds.x,residual<0.0),bounds.y,residual>0.0); }
                value=clamp(value,bounds.x,bounds.y);
            } else if !solved_bilateral(row,scratch[row_flags+i]) { value = max(value, 0.0); }
            scratch[impulse+i] = value;
        }
    }
    }
    var needs_projection=false;
    for (var i=0u;i<r;i++) {
        var residual=-scratch[rhs+i];
        for (var j=0u;j<r;j++) { residual+=scratch[gram+i*r+j]*scratch[impulse+j]; }
        let row=rows[island.row_start+i];
        let value=scratch[impulse+i];
        var error=abs(residual);
        if drive_row(row.kind) {
            let bounds=drive_bounds(row,scratch[search+i]);
            if bounds.x==bounds.y { error=0.0; }
            else if value==bounds.x { error=max(-residual,0.0); }
            else if value==bounds.y { error=max(residual,0.0); }
        } else if !solved_bilateral(row,scratch[row_flags+i]) && value==0.0 {
            error=max(-residual,0.0);
        }
        needs_projection=needs_projection || !finite(value) || !finite(residual) || error>1e-10;
    }
    if needs_projection {
        for (var i=0u;i<r;i++) { scratch[impulse+i]=0.0; }
        for (var sweep=0u;sweep<32768u;sweep++) {
            var changed=false;
            for (var i=0u;i<r;i++) {
                let row=rows[island.row_start+i];
                let diagonal=scratch[gram+i*r+i];
                var free=-scratch[rhs+i];
                for (var j=0u;j<r;j++) {
                    if j!=i { free+=scratch[gram+i*r+j]*scratch[impulse+j]; }
                }
                var value=0.0;
                if diagonal>0.0 { value=-free/diagonal; }
                if drive_row(row.kind) {
                    let bounds=drive_bounds(row,scratch[search+i]);
                    if diagonal<=0.0 { value=select(select(0.0,bounds.y,free<0.0),bounds.x,free>0.0); }
                    value=clamp(value,bounds.x,bounds.y);
                } else if !solved_bilateral(row,scratch[row_flags+i]) { value=max(value,0.0); }
                changed=changed || bitcast<u32>(scratch[impulse+i])!=bitcast<u32>(value);
                scratch[impulse+i]=value;
            }
            var converged=true;
            for (var i=0u;i<r;i++) {
                var residual=-scratch[rhs+i];
                for (var j=0u;j<r;j++) { residual+=scratch[gram+i*r+j]*scratch[impulse+j]; }
                let row=rows[island.row_start+i];
                let value=scratch[impulse+i];
                var error=abs(residual);
                if drive_row(row.kind) {
                    let bounds=drive_bounds(row,scratch[search+i]);
                    if bounds.x==bounds.y { error=0.0; }
                    else if value==bounds.x { error=max(-residual,0.0); }
                    else if value==bounds.y { error=max(residual,0.0); }
                } else if !solved_bilateral(row,scratch[row_flags+i]) && value==0.0 { error=max(-residual,0.0); }
                converged=converged && finite(value) && finite(residual) && error<=1e-10;
            }
            if converged || !changed { break; }
        }
    }
    if bilateral_only && needs_projection {
        minimum_norm_impulses(gram,r,impulse,residuals,search,product);
    }
    if nonlinear {
        for (var a=0u;a<island.art_count;a++) {
            let art=arts[island.art_start+a];
            for (var c=0u;c<art.width;c++) {
                var rate=velocities[art.global+c];
                for (var i=0u;i<r;i++) { rate+=scratch[response+i*n+art.local+c]*scratch[impulse+i]; }
                if !finite(rate) { atomicOr(&status,512u); return; }
                scratch[trial+art.local+c]=rate;
            }
        }
        prediction_converged=true;
        for (var i=0u;i<r;i++) {
            let row=rows[island.row_start+i];
            if euler_row(row.kind) && ((row.enabled.y>>1u)&7u)==0u {
                prediction_converged=prediction_converged && interval_feasible(row,trial,position_trial,4u*(island.row_start+i)+3u);
            }
        }
    }
    if prediction_converged { break; }
    }
    if !prediction_converged { atomicOr(&status,512u); return; }
    for (var i = 0u; i < r; i++) {
        var residual = -scratch[rhs+i];
        for (var j = 0u; j < r; j++) { residual += scratch[gram+i*r+j]*scratch[impulse+j]; }
        let kind = rows[island.row_start+i].kind;
        let value = scratch[impulse+i];
        let equality = solved_bilateral(rows[island.row_start+i],scratch[row_flags+i]);
        let tolerance = 2e-5*(1.0+abs(scratch[rhs+i]));
        var error=abs(residual);
        if drive_row(kind) {
            let bounds=drive_bounds(rows[island.row_start+i],scratch[search+i]);
            if bounds.x==bounds.y {error=0.0;}
            else if value==bounds.x {error=max(-residual,0.0);}
            else if value==bounds.y {error=max(residual,0.0);}
        } else if !equality && value==0.0 {error=max(-residual,0.0);}
        if !finite(residual) || !finite(value) || error>tolerance {
            atomicOr(&status,512u); return;
        }
        let row_id = island.row_start+i;
        let direction = vec3<f32>(scratch[directions+3u*i], scratch[directions+3u*i+1u], scratch[directions+3u*i+2u]);
        var sample = samples[row_id];
        let force = direction*(value/params.dt);
        for (var axis = 0u; axis < 3u; axis++) {
            atomicStore(&row_forces[4u*row_id+axis],bitcast<u32>(force[axis]));
        }
        atomicStore(&row_forces[4u*row_id+3u],0u);
        sample.force += vec4<f32>(direction*(value/params.dt)*params.weight, 0.0);
        if equality { sample.tag = vec4<u32>(3u, params.substep, 0u, 0u); }
        else if value != 0.0 { sample.tag = vec4<u32>(kind, params.substep, 0u, 0u); }
        samples[row_id] = sample;
    }
    for (var a = 0u; a < island.art_count; a++) {
        let art = arts[island.art_start+a];
        for (var c = 0u; c < art.width; c++) {
            var change = 0.0;
            for (var i = 0u; i < r; i++) { change += scratch[response+i*n+art.local+c]*scratch[impulse+i]; }
            var solved_delta = velocity_delta[art.global+c]+change;
            var rate=input_velocities[art.global+c]+solved_delta;
            for (var i=0u;i<r;i++) {
                let row=rows[island.row_start+i];
                if row.kind==15u && row.enabled.x==0u && row.bound==0.0 && row.b.w==0.0 && row.dof==art.local+c {
                    let roundoff=64.0*1.1920928955078125e-7*max(abs(input_velocities[art.global+c]),abs(velocities[art.global+c]));
                    if !finite(rate) || abs(rate)>1e-10+roundoff { atomicOr(&status,512u); return; }
                    rate=0.0;
                    solved_delta=-input_velocities[art.global+c];
                }
            }
            velocity_delta[art.global+c] = solved_delta;
            velocities[art.global+c] = rate;
        }
    }
}
