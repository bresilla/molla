const B: u32 = __BODIES__u;
const D: u32 = __DOFS__u;
struct Pose { position: vec4<f32>, rotation: vec4<f32> }
struct Body {
    metadata: vec4<u32>, properties: vec4<f32>, com: vec4<f32>,
    inertia0: vec4<f32>, inertia1: vec4<f32>, inertia2: vec4<f32>,
    parent_anchor: Pose, child_anchor: Pose, axis: vec4<f32>, gravity: vec4<f32>,
}
struct Spatial { linear: vec3<f64>, angular: vec3<f64> }
struct Load { linear: vec4<f32>, angular: vec4<f32> }
@group(0) @binding(0) var<storage, read> model: array<Body>;
@group(0) @binding(1) var<storage, read> poses: array<Pose>;
@group(0) @binding(2) var<storage, read> position_low: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> rates: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> loads: array<Load>;
@group(0) @binding(5) var<storage, read_write> bias: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> status: array<u32>;
@group(0) @binding(7) var<storage, read_write> mass_matrix: array<f32>;
var<private> velocities: array<Spatial, B>;
var<private> accelerations: array<Spatial, B>;
var<private> forces: array<Spatial, B>;
var<private> columns: array<Spatial, D>;
var<private> result: array<vec2<f32>, D>;
var<private> gram: array<f64, D * D>;
var<private> jacobian: array<Spatial, D>;

fn finite(value: f64) -> bool {
    return value == value && abs(value) <= f64(1.7976931348623157e308);
}
fn finite3(v: vec3<f64>) -> bool { return finite(v.x) && finite(v.y) && finite(v.z); }
fn rotation(q: vec4<f64>, v: vec3<f64>) -> vec3<f64> {
    return v + f64(2.0) * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn pose_vector(q: vec4<f64>, v: vec3<f64>) -> vec3<f64> {
    return f64(2.0)*dot(q.xyz,v)*q.xyz+(q.w*q.w-dot(q.xyz,q.xyz))*v
        +f64(2.0)*q.w*cross(q.xyz,v);
}
fn multiply(a: vec4<f64>, b: vec4<f64>) -> vec4<f64> {
    return vec4<f64>(a.w*b.xyz + b.w*a.xyz + cross(a.xyz,b.xyz), a.w*b.w-dot(a.xyz,b.xyz));
}
fn zero() -> Spatial { return Spatial(vec3<f64>(f64(0.0)), vec3<f64>(f64(0.0))); }
fn add(a: Spatial, b: Spatial) -> Spatial { return Spatial(a.linear+b.linear, a.angular+b.angular); }
fn scaled(a: Spatial, b: f64) -> Spatial { return Spatial(a.linear*b,a.angular*b); }
fn inertia(v: Spatial, m: f64, center: vec3<f64>, tensor: mat3x3<f64>) -> Spatial {
    let linear = m*(v.linear + cross(v.angular,center));
    return Spatial(linear,tensor*v.angular+cross(center,linear));
}
fn relative_position(body: u32, root: u32) -> vec3<f64> {
    return (vec3<f64>(poses[body].position.xyz)-vec3<f64>(poses[root].position.xyz))
        +(vec3<f64>(position_low[body].xyz)-vec3<f64>(position_low[root].xyz));
}
fn width(kind: u32) -> u32 {
    if kind == 4u { return 6u; }
    if kind == 3u { return 0u; }
    return 1u;
}

@compute @workgroup_size(1)
fn evaluate() {
    if status[0] != 0u { return; }
    let root = model[0].metadata.y;
    for (var k=0u; k<B; k++) {
        let body = model[k];
        let parent = body.metadata.x;
        let child = body.metadata.y;
        let kind = body.metadata.z;
        let offset = body.metadata.w;
        let q = vec4<f64>(poses[child].rotation);
        let point = relative_position(child,root);
        if !finite3(point) || !finite(dot(q,q)) || abs(dot(q,q)-f64(1.0))>f64(1e-4)
            || !finite3(vec3<f64>(loads[child].linear.xyz))
            || !finite3(vec3<f64>(loads[child].angular.xyz)) { status[0]=1u; return; }
        var frame_q = vec4<f64>(body.parent_anchor.rotation);
        var pivot = vec3<f64>(body.parent_anchor.position.xyz)
            -vec3<f64>(poses[root].position.xyz)-vec3<f64>(position_low[root].xyz);
        var vp = zero();
        var ap = Spatial(-vec3<f64>(body.gravity.xyz),vec3<f64>(f64(0.0)));
        if parent != 0xffffffffu {
            let p = model[parent].metadata.y;
            let pq = vec4<f64>(poses[p].rotation);
            pivot = relative_position(p,root) + pose_vector(pq,vec3<f64>(body.parent_anchor.position.xyz));
            frame_q = multiply(pq,frame_q);
            vp = velocities[parent];
            ap = accelerations[parent];
        }
        frame_q = normalize(frame_q);
        if kind == 4u { pivot = point + pose_vector(q,vec3<f64>(body.child_anchor.position.xyz)); }
        var vj = zero();
        var translation = vec3<f64>(f64(0.0));
        for (var d=0u; d<width(kind); d++) {
            var axis = vec3<f64>(body.axis.xyz);
            if kind == 4u {
                axis = vec3<f64>(f64(0.0));
                axis[d%3u] = f64(1.0);
            }
            axis = rotation(frame_q,axis);
            if kind<=1u { axis=normalize(axis); }
            var column = Spatial(axis,vec3<f64>(f64(0.0)));
            if kind == 1u || (kind == 4u && d>=3u) { column = Spatial(cross(axis,-pivot),axis); }
            columns[offset+d] = column;
            let rate = f64(rates[offset+d].x)+f64(rates[offset+d].y);
            if !finite(rate) { status[0]=1u; return; }
            vj = add(vj,scaled(column,rate));
            if kind == 4u && d<3u { translation += axis*rate; }
        }
        let v = add(vp,vj);
        var a = add(ap,Spatial(cross(v.angular,vj.linear)+cross(v.linear,vj.angular),cross(v.angular,vj.angular)));
        if kind == 4u { a.linear -= cross(vj.angular,translation); }
        velocities[k] = v;
        accelerations[k] = a;
        let center = point + pose_vector(q,vec3<f64>(body.com.xyz));
        let r = mat3x3<f64>(rotation(q,vec3<f64>(f64(1.0),f64(0.0),f64(0.0))),
            rotation(q,vec3<f64>(f64(0.0),f64(1.0),f64(0.0))),
            rotation(q,vec3<f64>(f64(0.0),f64(0.0),f64(1.0))));
        let tensor = r * mat3x3<f64>(vec3<f64>(body.inertia0.xyz),vec3<f64>(body.inertia1.xyz),vec3<f64>(body.inertia2.xyz)) * transpose(r);
        let momentum = inertia(v,f64(body.properties.x),center,tensor);
        var force = add(inertia(a,f64(body.properties.x),center,tensor),
            Spatial(cross(v.angular,momentum.linear),cross(v.angular,momentum.angular)+cross(v.linear,momentum.linear)));
        let load = vec3<f64>(loads[child].linear.xyz);
        force.linear -= load;
        force.angular -= vec3<f64>(loads[child].angular.xyz)+cross(center,load);
        if !finite3(force.linear) || !finite3(force.angular) { status[0]=2u; return; }
        forces[k] = force;
        for (var d=0u; d<D; d++) { jacobian[d]=zero(); }
        var ancestor=k;
        loop {
            let joint=model[ancestor];
            for (var d=0u; d<width(joint.metadata.z); d++) {
                jacobian[joint.metadata.w+d]=columns[joint.metadata.w+d];
            }
            ancestor=joint.metadata.x;
            if ancestor==0xffffffffu { break; }
        }
        for (var i=0u; i<D; i++) {
            let vi=jacobian[i].linear+cross(jacobian[i].angular,center);
            for (var j=0u; j<=i; j++) {
                let vj=jacobian[j].linear+cross(jacobian[j].angular,center);
                gram[i*D+j] += f64(body.properties.x)*dot(vi,vj)
                    +dot(jacobian[i].angular,tensor*jacobian[j].angular);
            }
        }
    }
    for (var reverse=0u; reverse<B; reverse++) {
        let k=B-1u-reverse;
        let body=model[k];
        let force=forces[k];
        for (var d=0u; d<width(body.metadata.z); d++) {
            let index=body.metadata.w+d;
            let value=dot(columns[index].linear,force.linear)+dot(columns[index].angular,force.angular);
            let high=f32(value);
            let low=f32(value-f64(high));
            if !finite(f64(high)) || !finite(f64(low)) { status[0]=2u; return; }
            result[index]=vec2<f32>(high,low);
        }
        if body.metadata.x != 0xffffffffu { forces[body.metadata.x]=add(forces[body.metadata.x],force); }
    }
    for (var i=0u; i<D; i++) {
        for (var j=0u; j<=i; j++) {
            if !finite(f64(f32(gram[i*D+j]))) { status[0]=2u; return; }
        }
    }
    for (var i=0u; i<D; i++) {
        bias[i]=result[i];
        for (var j=0u; j<=i; j++) {
            let value=f32(gram[i*D+j]);
            mass_matrix[i*D+j]=value;
            mass_matrix[j*D+i]=value;
        }
    }
}
