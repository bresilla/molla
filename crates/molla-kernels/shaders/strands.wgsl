struct Fiber {
    com_mass: vec4<f32>,
    inertia0: vec4<f32>, inertia1: vec4<f32>, inertia2: vec4<f32>,
    inverse0: vec4<f32>, inverse1: vec4<f32>, inverse2: vec4<f32>,
    vertices: vec4<u32>,
    surface: vec4<u32>,
}
struct Motion { center: vec4<f32>, rotation: vec4<f32>, velocity: vec4<f32>, omega: vec4<f32> }
struct Bond {
    fibers: vec4<u32>, anchor0: vec4<f32>, anchor1: vec4<f32>,
    rest_offset: vec4<f32>, rest_rotation: vec4<f32>,
    translation: vec4<f32>, rotation: vec4<f32>,
    force_limit: vec4<f32>, torque_limit: vec4<f32>, material: vec4<f32>,
}
struct BondState { lambda0: vec4<f32>, lambda1: vec4<f32>, damage: vec4<f32> }
struct Params { step: vec4<f32>, gravity: vec4<f32>, counts: vec4<u32>, coupled: vec4<f32>, numerics: vec4<f32>, grab_anchor: vec4<f32>, grab_target: vec4<f32>, grab_velocity: vec4<f32>, islands: vec4<u32>, tree: vec4<u32> }
struct Batch { offset: u32, count: u32, reverse: u32, pad: u32 }
struct Row { value: f32, compliance: f32, la: vec3<f32>, lb: vec3<f32>, aa: vec3<f32>, ab: vec3<f32> }
@group(0) @binding(0) var<storage, read> fibers: array<Fiber>;
@group(0) @binding(1) var<storage, read> bonds: array<Bond>;
@group(0) @binding(2) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> colors: array<u32>;
@group(0) @binding(4) var<storage, read_write> motions: array<Motion>;
@group(0) @binding(5) var<storage, read_write> predicted: array<Motion>;
@group(0) @binding(6) var<storage, read_write> bond_states: array<BondState>;
@group(0) @binding(7) var<storage, read_write> ground_lambda: array<f32>;
@group(0) @binding(8) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> live: array<Motion>;
@group(0) @binding(10) var<storage, read_write> live_bonds: array<BondState>;
@group(0) @binding(11) var<uniform> params: Params;
@group(0) @binding(12) var<uniform> batch: Batch;

fn finite4(v: vec4<f32>) -> bool { return all((bitcast<vec4<u32>>(v)&vec4<u32>(0x7f800000u)) != vec4<u32>(0x7f800000u)); }
fn conj(q: vec4<f32>) -> vec4<f32> { return vec4<f32>(-q.xyz,q.w); }
fn mul(a: vec4<f32>,b: vec4<f32>) -> vec4<f32> { return vec4<f32>(a.w*b.xyz+b.w*a.xyz+cross(a.xyz,b.xyz),a.w*b.w-dot(a.xyz,b.xyz)); }
fn rot(q: vec4<f32>,v: vec3<f32>) -> vec3<f32> { return v+2.0*cross(q.xyz,cross(q.xyz,v)+q.w*v); }
fn exponential(v: vec3<f32>) -> vec4<f32> {
    let t2 = dot(v,v);
    if t2 < 1e-8 { return normalize(vec4<f32>(v*(0.5-t2/48.0),1.0-t2/8.0)); }
    let t = sqrt(t2); return vec4<f32>(v*(sin(0.5*t)/t),cos(0.5*t));
}
fn logarithm(input: vec4<f32>) -> vec3<f32> {
    var q = normalize(input); if q.w < 0.0 { q = -q; }
    let s2 = dot(q.xyz,q.xyz);
    if s2 < 1e-10 { return q.xyz*(2.0+s2/3.0); }
    let s = sqrt(s2); return q.xyz*(2.0*atan2(s,q.w)/s);
}
fn inverse_apply(f: Fiber,q: vec4<f32>,v: vec3<f32>) -> vec3<f32> {
    return rot(q,mat3x3<f32>(f.inverse0.xyz,f.inverse1.xyz,f.inverse2.xyz)*rot(conj(q),v));
}
fn row(b: Bond,a: Motion,c: Motion,damage: f32,k: u32) -> Row {
    let fa = fibers[b.fibers.x]; let fb = fibers[b.fibers.y];
    let ra = rot(a.rotation,b.anchor0.xyz-fa.com_mass.xyz);
    let rb = rot(c.rotation,b.anchor1.xyz-fb.com_mass.xyz);
    let delta = c.center.xyz+rb-a.center.xyz-ra;
    var axis = vec3<f32>(0.0); axis[k%3u] = 1.0;
    let section = max(1.0-damage,1e-6);
    if k < 3u {
        let n = rot(a.rotation,axis);
        return Row((rot(conj(a.rotation),delta)-b.rest_offset.xyz)[k],b.translation[k]/section,
            -n,n,cross(n,delta)-cross(ra,n),cross(rb,n));
    }
    let r = logarithm(mul(mul(conj(a.rotation),c.rotation),conj(b.rest_rotation)));
    let t2 = dot(r,r); var term = 1.0/12.0+t2/720.0+t2*t2/30240.0;
    if t2 >= 1e-6 { let half = 0.5*sqrt(t2); term = (1.0-half/tan(half))/t2; }
    let n = rot(a.rotation,axis+0.5*cross(r,axis)+term*cross(r,cross(r,axis)));
    return Row(r[k-3u],b.rotation[k-3u]/section,vec3<f32>(0.0),vec3<f32>(0.0),-n,n);
}
fn weight(r: Row,a: Motion,b: Motion,fa: Fiber,fb: Fiber) -> f32 {
    return fa.com_mass.w*dot(r.la,r.la)+fb.com_mass.w*dot(r.lb,r.lb)
        +dot(r.aa,inverse_apply(fa,a.rotation,r.aa))+dot(r.ab,inverse_apply(fb,b.rotation,r.ab));
}
fn correction(m: Motion,f: Fiber,linear: vec3<f32>,angular: vec3<f32>,lambda: f32) -> Motion {
    if f.com_mass.w == 0.0 { return m; }
    var result = m; result.center = vec4<f32>(m.center.xyz+linear*(f.com_mass.w*lambda),0.0);
    result.rotation = normalize(mul(exponential(inverse_apply(f,m.rotation,angular)*lambda),m.rotation));
    return result;
}
@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x; if i >= params.counts.x || atomicLoad(&status[0]) != 0u { return; }
    if i == 0u { atomicStore(&status[1],0u); atomicStore(&status[2],0u); atomicStore(&status[3],0u); atomicStore(&status[4],0u); }
    let f = fibers[i]; var m = live[i];
    for (var j = 0u; j < f.vertices.y; j++) { ground_lambda[f.vertices.x+j] = 0.0; }
    if f.com_mass.w > 0.0 {
        let dt = params.step.x; let attenuation = exp(-params.step.y*dt);
        m.velocity = vec4<f32>((m.velocity.xyz+params.gravity.xyz*dt)*attenuation,0.0);
        let q = m.rotation;
        let inertia = mat3x3<f32>(f.inertia0.xyz,f.inertia1.xyz,f.inertia2.xyz);
        let inv = mat3x3<f32>(f.inverse0.xyz,f.inverse1.xyz,f.inverse2.xyz);
        let initial = rot(conj(q),m.omega.xyz)*attenuation;
        let momentum = rot(q,inertia*initial);
        let half = 0.5*dt;
        var omega = initial;
        for (var j = 0u; j < 12u; j++) {
            let local_momentum = inertia*omega;
            let residual = omega-initial+half*(inv*cross(omega,local_momentum));
            if length(residual) <= 2e-7*(1.0+length(initial)) { break; }
            var jacobian: mat3x3<f32>;
            for (var k = 0u; k < 3u; k++) {
                var axis = vec3<f32>(0.0); axis[k] = 1.0;
                jacobian[k] = axis+half*(inv*(cross(axis,local_momentum)+cross(omega,inertia[k])));
            }
            let a = jacobian[0]; let b = jacobian[1]; let c = jacobian[2];
            omega -= vec3<f32>(dot(cross(b,c),residual),dot(cross(c,a),residual),dot(cross(a,b),residual))/dot(a,cross(b,c));
        }
        let residual = omega-initial+half*(inv*cross(omega,inertia*omega));
        if !finite4(vec4<f32>(omega,length(residual))) || length(residual) > 1e-5*(1.0+length(initial)) {
            atomicOr(&status[0],16u); return;
        }
        m.center = vec4<f32>(m.center.xyz+m.velocity.xyz*dt,0.0);
        m.rotation = normalize(mul(q,normalize(vec4<f32>(omega*half,1.0))));
        m.omega = vec4<f32>(inverse_apply(f,m.rotation,momentum),0.0);
    }
    motions[i] = m; predicted[i] = m;
    predicted[i].center.w = 0.0; predicted[i].velocity.w = 0.0; predicted[i].omega.w = 0.0;
}
fn grab_parameters(i: u32) -> vec4<f32> {
    let ratio = params.grab_velocity.w*params.grab_anchor.w/params.step.x;
    let previous = live[i].center.xyz+rot(live[i].rotation,params.grab_anchor.xyz-fibers[i].com_mass.xyz);
    let goal = (params.grab_target.xyz+ratio*(previous+params.step.x*params.grab_velocity.xyz))/(1.0+ratio);
    var compliance = params.grab_anchor.w/(1.0+ratio);
    if params.numerics.z > 0.0 {
        let point = motions[i].center.xyz+rot(motions[i].rotation,params.grab_anchor.xyz-fibers[i].com_mass.xyz);
        compliance = max(compliance,distance(point,goal)/params.numerics.z);
    }
    return vec4<f32>(goal,compliance);
}
@compute @workgroup_size(64)
fn project_grab(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u || params.grab_target.w < 0.0 || atomicLoad(&status[0]) != 0u { return; }
    let i = u32(params.grab_target.w); let f = fibers[i]; var m = motions[i];
    var lambda = vec3<f32>(predicted[i].center.w,predicted[i].velocity.w,predicted[i].omega.w);
    let grab = grab_parameters(i); let alpha = grab.w/(params.step.x*params.step.x);
    for (var k = 0u; k < 3u; k++) {
        var axis = vec3<f32>(0.0); axis[k] = 1.0;
        let r = rot(m.rotation,params.grab_anchor.xyz-f.com_mass.xyz);
        let error = dot(m.center.xyz+r-grab.xyz,axis); let angular = cross(r,axis);
        let inv = inverse_apply(f,m.rotation,angular); let w = f.com_mass.w+dot(angular,inv);
        var dl = (-error-alpha*lambda[k])/(w+alpha); let angle = length(inv)*abs(dl);
        if angle > 0.2 { dl *= 0.2/angle; }
        lambda[k] += dl; m = correction(m,f,axis,angular,dl);
    }
    predicted[i].center.w = lambda.x; predicted[i].velocity.w = lambda.y; predicted[i].omega.w = lambda.z;
    motions[i] = m;
}
@compute @workgroup_size(64)
fn initialize_bonds(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x; if i >= params.counts.y || atomicLoad(&status[0]) != 0u { return; }
    bond_states[i] = BondState(vec4<f32>(0.0),vec4<f32>(0.0),live_bonds[i].damage);
}
@compute @workgroup_size(64)
fn project_bonds(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= batch.count || atomicLoad(&status[0]) != 0u { return; }
    let i = colors[batch.offset+id.x]; let b = bonds[i]; var state = bond_states[i];
    if state.damage.x >= 1.0 { return; }
    let ai = b.fibers.x; let bi = b.fibers.y; let fa = fibers[ai]; let fb = fibers[bi];
    var a = motions[ai]; var c = motions[bi];
    for (var k = 0u; k < 6u; k++) {
        let axis = select(k,5u-k,batch.reverse != 0u); let r = row(b,a,c,state.damage.x,axis);
        let w = weight(r,a,c,fa,fb); if w == 0.0 { continue; }
        let alpha = r.compliance/(params.step.x*params.step.x);
        var lambda = 0.0; if axis < 3u { lambda = state.lambda0[axis]; } else { lambda = state.lambda1[axis-3u]; }
        let dl = (-r.value-alpha*lambda)/(w+alpha);
        if axis < 3u { state.lambda0[axis] += dl; } else { state.lambda1[axis-3u] += dl; }
        a = correction(a,fa,r.la,r.aa,dl); c = correction(c,fb,r.lb,r.ab,dl);
    }
    motions[ai] = a; motions[bi] = c; bond_states[i] = state;
}
@compute @workgroup_size(64)
fn project_ground(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x; if i >= params.counts.x || params.counts.z == 0u || atomicLoad(&status[0]) != 0u { return; }
    let f = fibers[i]; if f.com_mass.w == 0.0 { return; }
    var m = motions[i]; let n = vec3<f32>(0.0,1.0,0.0);
    for (var j = 0u; j < f.vertices.y; j++) {
    let index = f.vertices.x+j;
    let point = m.center.xyz+rot(m.rotation,vertices[index].xyz-f.com_mass.xyz);
    if point.y >= params.step.z && ground_lambda[index] == 0.0 { continue; }
    let angular = cross(point-m.center.xyz,n);
    let w = f.com_mass.w+dot(angular,inverse_apply(f,m.rotation,angular));
    let lambda = max(0.0,ground_lambda[index]-(point.y-params.step.z)/w);
    var dl = lambda-ground_lambda[index]; let angle = length(inverse_apply(f,m.rotation,angular))*abs(dl);
    if angle > 0.2 { dl *= 0.2/angle; }
    m = correction(m,f,n,angular,dl); ground_lambda[index] += dl;
    }
    motions[i] = m;
}
@compute @workgroup_size(64)
fn recover(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x; if i >= params.counts.x || atomicLoad(&status[0]) != 0u || fibers[i].com_mass.w == 0.0 { return; }
    var m = motions[i]; let p = predicted[i];
    m.velocity = vec4<f32>(m.velocity.xyz+(m.center.xyz-p.center.xyz)/params.step.x,0.0);
    m.omega = vec4<f32>(m.omega.xyz+logarithm(mul(m.rotation,conj(p.rotation)))/params.step.x,0.0);
    motions[i] = m;
}
@compute @workgroup_size(64)
fn damage_bonds(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x; if i >= params.counts.y || atomicLoad(&status[0]) != 0u { return; }
    var s = bond_states[i]; if s.damage.x >= 1.0 { return; }
    let b = bonds[i]; let section = max(1.0-s.damage.x,1e-6); let dt2 = params.step.x*params.step.x;
    let force = s.lambda0.xyz/dt2; let torque = s.lambda1.xyz/dt2;
    let utilization = max(length(force/(b.force_limit.xyz*section)),length(torque/(b.torque_limit.xyz*section)));
    if !finite4(vec4<f32>(force,utilization)) || !finite4(vec4<f32>(torque,0.0)) { atomicOr(&status[0],2u); return; }
    s.damage.x = min(1.0,s.damage.x+params.step.x*b.material.x*max(utilization-1.0,0.0)); bond_states[i] = s;
}
@compute @workgroup_size(64)
fn damp_bonds(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= batch.count || atomicLoad(&status[0]) != 0u { return; }
    let i = colors[batch.offset+id.x]; let b = bonds[i]; let s = bond_states[i]; if s.damage.x >= 1.0 { return; }
    let ai = b.fibers.x; let bi = b.fibers.y; let fa = fibers[ai]; let fb = fibers[bi];
    var a = motions[ai]; var c = motions[bi]; let factor = 1.0-exp(-b.material.y*params.step.x);
    for (var k = 0u; k < 6u; k++) {
        let r = row(b,a,c,s.damage.x,k); let w = weight(r,a,c,fa,fb); if w == 0.0 { continue; }
        let rate = dot(r.la,a.velocity.xyz)+dot(r.lb,c.velocity.xyz)+dot(r.aa,a.omega.xyz)+dot(r.ab,c.omega.xyz);
        let impulse = -factor*rate/w;
        a.velocity += vec4<f32>(r.la*(fa.com_mass.w*impulse),0.0); c.velocity += vec4<f32>(r.lb*(fb.com_mass.w*impulse),0.0);
        a.omega += vec4<f32>(inverse_apply(fa,a.rotation,r.aa)*impulse,0.0); c.omega += vec4<f32>(inverse_apply(fb,c.rotation,r.ab)*impulse,0.0);
    }
    motions[ai] = a; motions[bi] = c;
}
@compute @workgroup_size(64)
fn friction(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x; if i >= params.counts.x || atomicLoad(&status[0]) != 0u { return; }
    let f = fibers[i]; if f.com_mass.w == 0.0 || params.step.w == 0.0 { return; }
    var m = motions[i];
    for (var j = 0u; j < f.vertices.y; j++) {
    let index = f.vertices.x+j; let lambda = ground_lambda[index]; if lambda == 0.0 { continue; }
    let r = rot(m.rotation,vertices[index].xyz-f.com_mass.xyz);
    let velocity = m.velocity.xyz+cross(m.omega.xyz,r); let tangent = vec3<f32>(velocity.x,0.0,velocity.z);
    if dot(tangent,tangent) == 0.0 { continue; }
    let n = normalize(tangent); let angular = cross(r,n); let w = f.com_mass.w+dot(angular,inverse_apply(f,m.rotation,angular));
    let impulse = -n*min(length(tangent)/w,params.step.w*lambda/params.step.x);
    m.velocity += vec4<f32>(impulse*f.com_mass.w,0.0); m.omega += vec4<f32>(inverse_apply(f,m.rotation,cross(r,impulse)),0.0);
    }
    motions[i] = m;
}
@compute @workgroup_size(64)
fn validate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i < params.counts.x {
        let m = motions[i];
        if !finite4(m.center) || !finite4(m.rotation) || !finite4(m.velocity) || !finite4(m.omega)
            || abs(dot(m.rotation,m.rotation)-1.0) > 1e-3 { atomicOr(&status[0],1u); }
    }
    if i < params.counts.y {
        let b = bond_states[i];
        if !finite4(b.lambda0) || !finite4(b.lambda1) || !finite4(b.damage) || b.damage.x < 0.0 || b.damage.x > 1.0 { atomicOr(&status[0],2u); }
    }
}
@compute @workgroup_size(64)
fn publish(@builtin(global_invocation_id) id: vec3<u32>) {
    if atomicLoad(&status[0]) != 0u { return; }
    let i = id.x; if i < params.counts.x { live[i] = motions[i]; } if i < params.counts.y { live_bonds[i] = bond_states[i]; }
}
