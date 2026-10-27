struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct Wrench { linear: vec4<f32>, angular: vec4<f32> }
struct Spatial { linear: vec3<f32>, angular: vec3<f32> }
struct Joint {
    parent: i32, child: u32, kind: u32, dof: u32,
    coordinate: u32, columns: u32, reversed: u32, padding: u32,
    parent_frame: Transform, child_frame: Transform,
    axis: vec4<f32>, mass: vec4<f32>,
    inertia: mat3x4<f32>, com: vec4<f32>,
}
struct Params { dt: f32, count: u32, correction: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> articulations: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> joints: array<Joint>;
@group(0) @binding(2) var<storage, read> input_pose: array<Transform>;
@group(0) @binding(3) var<storage, read> input_velocity: array<Wrench>;
@group(0) @binding(4) var<storage, read> input_rate: array<f32>;
@group(0) @binding(5) var<storage, read> input_low: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> output_pose: array<Transform>;
@group(0) @binding(7) var<storage, read_write> output_velocity: array<Wrench>;
@group(0) @binding(8) var<storage, read_write> output_rate: array<f32>;
@group(0) @binding(9) var<storage, read_write> output_coordinate: array<f32>;
@group(0) @binding(10) var<storage, read_write> output_low: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> coordinate_low: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read> projected_pose: array<Transform>;
@group(0) @binding(13) var<storage, read> projected_low: array<vec4<f32>>;
@group(0) @binding(14) var<storage, read> delta: array<f32>;
@group(0) @binding(15) var<storage, read> active_mask: array<u32>;
@group(0) @binding(16) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(17) var<uniform> params: Params;
@group(0) @binding(18) var<storage, read> response_delta: array<f32>;

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w*b.xyz + b.w*a.xyz + cross(a.xyz,b.xyz), a.w*b.w - dot(a.xyz,b.xyz));
}
fn pose(body: u32, mode: u32) -> Transform {
    if mode == 0u { return input_pose[body]; }
    if mode == 1u { return projected_pose[body]; }
    return output_pose[body];
}
fn low(body: u32, mode: u32) -> vec3<f32> {
    if mode == 0u { return input_low[body].xyz; }
    if mode == 1u { return projected_low[body].xyz; }
    return vec3<f32>(bitcast<f32>(atomicLoad(&output_low[4u*body])),
        bitcast<f32>(atomicLoad(&output_low[4u*body+1u])),
        bitcast<f32>(atomicLoad(&output_low[4u*body+2u])));
}
fn relative(body: u32, origin: u32, mode: u32) -> vec3<f32> {
    return (pose(body,mode).position.xyz - input_pose[origin].position.xyz)
        + (low(body,mode) - input_low[origin].xyz);
}
fn center(j: Joint, origin: u32, mode: u32) -> vec3<f32> {
    return relative(j.child, origin, mode) + rotate(pose(j.child,mode).rotation,j.com.xyz);
}
fn center_delta(a: Joint, a_mode: u32, b: Joint, b_mode: u32) -> vec3<f32> {
    return (pose(a.child,a_mode).position.xyz-pose(b.child,b_mode).position.xyz)
        + (low(a.child,a_mode)-low(b.child,b_mode))
        + (rotate(pose(a.child,a_mode).rotation,a.com.xyz)-rotate(pose(b.child,b_mode).rotation,b.com.xyz));
}
fn frame(j: Joint, begin: u32, origin: u32, mode: u32) -> Transform {
    var position = (j.parent_frame.position.xyz - input_pose[origin].position.xyz) - input_low[origin].xyz;
    var rotation = normalize(j.parent_frame.rotation);
    if j.parent >= 0 {
        let parent = joints[begin+u32(j.parent)].child;
        let p = pose(parent,mode);
        position = relative(parent,origin,mode) + rotate(p.rotation,j.parent_frame.position.xyz);
        rotation = normalize(multiply(p.rotation,rotation));
    }
    let child = pose(j.child,mode);
    let child_position = relative(j.child,origin,mode) + rotate(child.rotation,j.child_frame.position.xyz);
    let child_rotation = normalize(multiply(child.rotation,j.child_frame.rotation));
    let parent_position = position;
    if j.reversed != 0u {
        position = child_position;
        rotation = child_rotation;
    }
    if j.kind >= 4u && j.kind <= 6u {
        position = child_position;
        if j.reversed != 0u { position = parent_position; }
    }
    return Transform(vec4<f32>(position,0.0),rotation);
}
fn width(kind: u32) -> u32 {
    if kind <= 1u { return 1u; }
    if kind == 2u { return 3u; }
    if kind >= 4u && kind <= 6u { return 6u; }
    if kind == 7u { return 2u; }
    return 0u;
}
fn column(j: Joint, d: u32, f: Transform) -> Spatial {
    if (j.columns & (1u << d)) == 0u { return Spatial(vec3<f32>(0.0),vec3<f32>(0.0)); }
    var local = vec3<f32>(0.0);
    var angular = false;
    if j.kind <= 1u { local = j.axis.xyz; angular = j.kind == 1u; }
    else if j.kind == 2u { local[d] = 1.0; angular = true; }
    else if j.kind >= 4u && j.kind <= 6u { local[d % 3u] = 1.0; angular = d >= 3u; }
    else if j.kind == 7u {
        local = select(vec3<f32>(0.0,0.0,1.0),vec3<f32>(1.0,0.0,0.0),d == 1u);
        angular = d == 1u;
    }
    let axis = normalize(rotate(f.rotation,local)) * select(1.0,-1.0,j.reversed != 0u);
    if angular { return Spatial(cross(axis,-f.position.xyz),axis); }
    return Spatial(axis,vec3<f32>(0.0));
}
fn inertia(j: Joint, mode: u32) -> mat3x3<f32> {
    let q = pose(j.child,mode).rotation;
    let r = mat3x3<f32>(rotate(q,vec3<f32>(1.0,0.0,0.0)),
        rotate(q,vec3<f32>(0.0,1.0,0.0)),rotate(q,vec3<f32>(0.0,0.0,1.0)));
    return r * mat3x3<f32>(j.inertia[0].xyz,j.inertia[1].xyz,j.inertia[2].xyz) * transpose(r);
}
fn inertial(mass: f32, c: vec3<f32>, i: mat3x3<f32>, v: Spatial) -> Spatial {
    let p = mass * (v.linear + cross(v.angular,c));
    return Spatial(p,i*v.angular + cross(c,p));
}
fn finite(v: vec3<f32>) -> bool { return all(abs(v) <= vec3<f32>(3.402823e38)); }
fn identity() -> mat3x3<f32> {
    return mat3x3<f32>(vec3<f32>(1.0,0.0,0.0),vec3<f32>(0.0,1.0,0.0),vec3<f32>(0.0,0.0,1.0));
}
fn solve(inertia_com: mat3x3<f32>, rhs: vec3<f32>) -> vec3<f32> {
    let scale = max(max(length(inertia_com[0]),length(inertia_com[1])),length(inertia_com[2]));
    if scale == 0.0 { return vec3<f32>(0.0); }
    var a = inertia_com * (1.0 / scale);
    var vectors = identity();
    for (var sweep = 0u; sweep < 24u; sweep += 1u) {
        var p = 0u; var q = 1u;
        if abs(a[0][2]) > abs(a[p][q]) { p = 0u; q = 2u; }
        if abs(a[1][2]) > abs(a[p][q]) { p = 1u; q = 2u; }
        if abs(a[p][q]) <= 1e-7 { break; }
        let angle = 0.5 * atan2(2.0*a[p][q],a[q][q]-a[p][p]);
        let c = cos(angle); let s = sin(angle);
        var r = identity();
        r[p][p] = c; r[q][q] = c; r[p][q] = -s; r[q][p] = s;
        a = transpose(r)*a*r;
        vectors = vectors*r;
    }
    var answer = vec3<f32>(0.0);
    for (var k = 0u; k < 3u; k += 1u) {
        if a[k][k] > 1e-7 { answer += vectors[k] * (dot(vectors[k],rhs) / (a[k][k]*scale)); }
    }
    return answer;
}
fn rounded(index: u32, value: f32, coordinate: bool) -> f32 {
    if coordinate {
        atomicStore(&coordinate_low[index],bitcast<u32>(value));
        return bitcast<f32>(atomicLoad(&coordinate_low[index]));
    }
    atomicStore(&output_low[index],bitcast<u32>(value));
    return bitcast<f32>(atomicLoad(&output_low[index]));
}
fn two_sum(index: u32, a: f32, b: f32, coordinate: bool) -> vec2<f32> {
    let sum = rounded(index,a+b,coordinate);
    let vb = rounded(index,sum-a,coordinate);
    let va = rounded(index,sum-vb,coordinate);
    return vec2<f32>(sum,rounded(index,(a-va)+(b-vb),coordinate));
}
fn advance(index: u32, high: f32, increment: f32, coordinate: bool) -> f32 {
    var residue = 0.0;
    if coordinate { residue = bitcast<f32>(atomicLoad(&coordinate_low[index])); }
    else { residue = bitcast<f32>(atomicLoad(&output_low[index])); }
    let step = two_sum(index,increment,residue,coordinate);
    let next = two_sum(index,high,step.x,coordinate);
    let normalized = two_sum(index,next.x,next.y+step.y,coordinate);
    _ = rounded(index,normalized.y,coordinate);
    return normalized.x;
}

@compute @workgroup_size(32)
fn transport(@builtin(global_invocation_id) gid: vec3<u32>) {
    let art = gid.x;
    if art >= params.count || atomicLoad(&status) != 0u { return; }
    let range = articulations[art];
    if range.z == 0u || active_mask[art] == 0u { return; }
    let origin = joints[range.x].child;
    var velocity: array<Spatial,__ART_WIDTH__>;
    var acceleration: array<Spatial,__ART_WIDTH__>;
    var corrected: array<Spatial,__ART_WIDTH__>;
    var projected: array<Spatial,__ART_WIDTH__>;
    var target_p = vec3<f32>(0.0); var target_h = vec3<f32>(0.0);
    var output_p = vec3<f32>(0.0); var output_h = vec3<f32>(0.0);
    var reference_offset = vec3<f32>(0.0); var output_offset = vec3<f32>(0.0);
    let root = joints[range.x];
    let output_anchor = center(root,origin,2u);
    var total_mass = 0.0;
    for (var index = range.x; index < range.y; index += 1u) {
        let k = index-range.x;
        let j = joints[index];
        let f = frame(j,range.x,origin,0u);
        let fp = frame(j,range.x,origin,1u);
        var vj = Spatial(vec3<f32>(0.0),vec3<f32>(0.0));
        var aj = vj; var vc = vj; var vr = vj;
        for (var d = 0u; d < width(j.kind); d += 1u) {
            let s = column(j,d,f); let sp = column(j,d,fp);
            let rate = input_rate[j.dof+d]; let next = output_rate[j.dof+d];
            vj.linear += s.linear*rate; vj.angular += s.angular*rate;
            var change = delta[j.dof+d];
            if params.correction != 0u { change += response_delta[j.dof+d]; }
            aj.linear += s.linear*change; aj.angular += s.angular*change;
            vc.linear += s.linear*next; vc.angular += s.angular*next;
            vr.linear += sp.linear*next; vr.angular += sp.angular*next;
        }
        var vi = vj; var ai = aj;
        if j.parent >= 0 {
            let p = u32(j.parent);
            vi.linear += velocity[p].linear; vi.angular += velocity[p].angular;
            ai.linear += acceleration[p].linear; ai.angular += acceleration[p].angular;
            vc.linear += corrected[p].linear; vc.angular += corrected[p].angular;
            vr.linear += projected[p].linear; vr.angular += projected[p].angular;
        }
        ai.linear += params.dt*(cross(vi.angular,vj.linear)+cross(vi.linear,vj.angular));
        ai.angular += params.dt*cross(vi.angular,vj.angular);
        if j.kind >= 4u && j.kind <= 6u {
            let translation = rotate(f.rotation,vec3<f32>(input_rate[j.dof],input_rate[j.dof+1u],input_rate[j.dof+2u]));
            ai.linear -= params.dt*cross(vj.angular,translation);
        }
        velocity[k] = vi; acceleration[k] = ai; corrected[k] = vc; projected[k] = vr;
        let c = center(j,origin,0u); let cp = center(j,origin,1u); let co = center(j,origin,2u);
        let ii = inertia(j,0u); let ip = inertia(j,1u); let io = inertia(j,2u);
        let mv = inertial(j.mass.x,c,ii,vi); let ma = inertial(j.mass.x,c,ii,ai);
        let mc = inertial(j.mass.x,c,ii,vc); let mr = inertial(j.mass.x,cp,ip,vr);
        let initial = input_velocity[j.child]; let final_velocity = output_velocity[j.child];
        let p0 = j.mass.x*initial.linear.xyz; let pn = j.mass.x*final_velocity.linear.xyz;
        target_p += p0 + ma.linear + params.dt*cross(vi.angular,mv.linear) + (mr.linear-mc.linear);
        target_h += ii*initial.angular.xyz + cross(c,p0) + ma.angular
            + params.dt*(cross(vi.angular,mv.angular)+cross(vi.linear,mv.linear)) + (mr.angular-mc.angular);
        output_p += pn; output_h += io*final_velocity.angular.xyz + cross(co,pn);
        total_mass += j.mass.x;
        reference_offset += center_delta(j,1u,root,1u)*j.mass.x;
        output_offset += center_delta(j,2u,root,2u)*j.mass.x;
    }
    reference_offset /= total_mass;
    output_offset /= total_mass;
    let output_center = output_anchor + output_offset;
    let shift = center_delta(root,1u,root,2u) + (reference_offset-output_offset)
        + params.dt*target_p/total_mass;
    let linear_delta = target_p-output_p;
    let angular_delta = target_h-output_h-cross(shift,output_p)-cross(output_center+shift,linear_delta);
    var composite = mat3x3<f32>();
    for (var index = range.x; index < range.y; index += 1u) {
        let j = joints[index]; let r = center_delta(j,2u,root,2u)-output_offset;
        composite += inertia(j,2u) + j.mass.x*(identity()*dot(r,r)-mat3x3<f32>(r*r.x,r*r.y,r*r.z));
    }
    let spin = solve(composite,angular_delta);
    let residual = composite*spin-angular_delta;
    let scale = length(target_h)+length(output_h)+length(output_center+shift)*(length(target_p)+length(output_p));
    if !(total_mass > 0.0 && total_mass <= 3.402823e38) || !finite(shift) || !finite(spin)
        || !finite(residual) || length(residual) > 3e-5*scale {
        atomicOr(&status,1u); return;
    }
    let f = frame(root,range.x,origin,2u);
    let inverse = vec4<f32>(-f.rotation.xyz,f.rotation.w);
    let direction = select(1.0,-1.0,root.reversed != 0u);
    let translation = linear_delta/total_mass + cross(spin,f.position.xyz-output_center);
    let root_linear = rotate(inverse,translation)*direction; let root_angular = rotate(inverse,spin)*direction;
    let coordinate_shift = rotate(inverse,shift)*direction;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        output_rate[root.dof+axis] += root_linear[axis];
        output_rate[root.dof+3u+axis] += root_angular[axis];
        let c = root.coordinate+axis;
        output_coordinate[c] = advance(c,output_coordinate[c],coordinate_shift[axis],true);
    }
    for (var index = range.x; index < range.y; index += 1u) {
        let j = joints[index]; let c = center(j,origin,2u);
        output_velocity[j.child].linear += vec4<f32>(linear_delta/total_mass+cross(spin,c-output_center),0.0);
        output_velocity[j.child].angular += vec4<f32>(spin,0.0);
        for (var axis = 0u; axis < 3u; axis += 1u) {
            let high = output_pose[j.child].position[axis];
            output_pose[j.child].position[axis] = advance(4u*j.child+axis,high,shift[axis],false);
        }
        if !finite(output_velocity[j.child].linear.xyz) || !finite(output_velocity[j.child].angular.xyz)
            || !finite(output_pose[j.child].position.xyz) { atomicOr(&status,1u); }
    }
    if !finite(vec3<f32>(output_rate[root.dof],output_rate[root.dof+1u],output_rate[root.dof+2u]))
        || !finite(vec3<f32>(output_rate[root.dof+3u],output_rate[root.dof+4u],output_rate[root.dof+5u])) {
        atomicOr(&status,1u);
    }
}
