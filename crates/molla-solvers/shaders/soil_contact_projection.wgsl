struct Node { arm_mass: vec4<f32>, momentum: vec4<f32> }
struct Moments { a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32> }
struct Params { width: u32, bodies: u32, nodes: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> spans: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> mass: array<f32>;
@group(0) @binding(3) var<storage, read> jacobian: array<f32>;
@group(0) @binding(4) var<storage, read> free_rates: array<f32>;
@group(0) @binding(5) var<storage, read_write> moments: array<Moments>;
@group(0) @binding(6) var<storage, read_write> system: array<f32>;
@group(0) @binding(7) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(8) var<storage, read_write> rates: array<f32>;
@group(0) @binding(9) var<storage, read_write> motion: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> impulse: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(12) var<uniform> params: Params;
@group(0) @binding(13) var<storage, read_write> boundary_impulse: array<vec4<f32>>;

var<workgroup> sums: array<Moments,64>;
var<workgroup> boundary_flags: array<u32,64>;
fn finite(v: f32) -> bool { return abs(v) <= 3.402823e38; }
fn finite3(v: vec3<f32>) -> bool { return all(abs(v) <= vec3<f32>(3.402823e38)); }
fn column(body: u32, row: u32, col: u32) -> vec3<f32> {
    let start = (body*6u+row)*params.width+col;
    return vec3<f32>(jacobian[start],jacobian[start+params.width],jacobian[start+2u*params.width]);
}
fn inertia(m: Moments, w: vec3<f32>) -> vec3<f32> {
    return mat3x3<f32>(vec3<f32>(m.b.x,m.b.w,m.c.x),
        vec3<f32>(m.b.w,m.b.y,m.c.y),vec3<f32>(m.c.x,m.c.y,m.b.z))*w;
}
fn linear_momentum(m: Moments) -> vec3<f32> { return vec3<f32>(m.c.z,m.c.w,m.d.x); }

@compute @workgroup_size(64)
fn reduce(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let body = group.x;
    let span = spans[body];
    var sum = Moments(vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0));
    var flags=0u;
    let valid = span.x <= params.nodes && span.y <= params.nodes-min(span.x,params.nodes);
    if !valid { atomicOr(&status,1u); }
    if body>0u {
        let previous=spans[body-1u];
        if previous.x>span.x || previous.y>span.x-min(previous.x,span.x) { atomicOr(&status,1u); }
    }
    if valid {
        for (var i=lane; i<span.y; i+=64u) {
            let node = nodes[span.x+i];
            let r = node.arm_mass.xyz; let m = node.arm_mass.w; let p = node.momentum.xyz;
            if !(node.momentum.w>=0.0 && node.momentum.w<=63.0) || floor(node.momentum.w)!=node.momentum.w {
                atomicOr(&status,1u); continue;
            }
            if params.pad==0u && node.momentum.w!=0.0 { atomicOr(&status,1u); continue; }
            if !finite3(r) || !finite3(p) || !finite(m) || m<0.0 || (m==0.0 && any(p!=vec3<f32>(0.0))) {
                atomicOr(&status,1u); continue;
            }
            if m>0.0 { flags|=u32(node.momentum.w); }
            sum.a += vec4<f32>(m,m*r);
            sum.b += m*vec4<f32>(r.y*r.y+r.z*r.z,r.x*r.x+r.z*r.z,r.x*r.x+r.y*r.y,-r.x*r.y);
            sum.c += vec4<f32>(-m*r.x*r.z,-m*r.y*r.z,p.x,p.y);
            sum.d += vec4<f32>(p.z,cross(r,p));
        }
    }
    sums[lane]=sum;
    boundary_flags[lane]=flags;
    workgroupBarrier();
    for (var width=32u; width>0u; width/=2u) {
        if lane<width {
            sums[lane].a+=sums[lane+width].a; sums[lane].b+=sums[lane+width].b;
            sums[lane].c+=sums[lane+width].c; sums[lane].d+=sums[lane+width].d;
            boundary_flags[lane]|=boundary_flags[lane+width];
        }
        workgroupBarrier();
    }
    if lane==0u {
        moments[body]=sums[0];
        boundary_impulse[2u*body]=vec4<f32>(0.0,0.0,0.0,f32(boundary_flags[0]));
        boundary_impulse[2u*body+1u]=vec4<f32>(0.0);
    }
}

@compute @workgroup_size(64)
fn assemble(@builtin(global_invocation_id) id: vec3<u32>) {
    let n=params.width; let index=id.x;
    if index>=n*n || atomicLoad(&status)!=0u { return; }
    let row=index/n; let col=index%n;
    var value=mass[index]; var force=0.0;
    if col==0u {
        for (var k=0u;k<n;k++) { force+=mass[row*n+k]*free_rates[k]; }
    }
    for (var b=0u;b<params.bodies;b++) {
        let m=moments[b];
        let vr=column(b,0u,row); let wr=column(b,3u,row);
        let vc=column(b,0u,col); let wc=column(b,3u,col);
        value+=dot(vr,m.a.x*vc-cross(m.a.yzw,wc))+dot(wr,cross(m.a.yzw,vc)+inertia(m,wc));
        if col==0u { force+=dot(vr,linear_momentum(m))+dot(wr,m.d.yzw); }
    }
    if !finite(value) || !finite(force) { atomicOr(&status,4u); }
    system[index]=value;
    if col==0u { rhs[row]=force; }
}

@compute @workgroup_size(1)
fn solve() {
    if atomicLoad(&status)!=0u { return; }
    let n=params.width;
    var lower: array<f32,1024>;
    var answer: array<f32,32>;
    var empty=true;
    for (var b=0u;b<params.bodies;b++) { empty=empty && moments[b].a.x==0.0; }
    for (var stage=0u;stage<select(2u,1u,empty);stage++) {
        var scale=0.0;
        for (var i=0u;i<n*n;i++) {
            var value=mass[i]; if stage==1u { value=system[i]; }
            if !finite(value) { atomicOr(&status,4u); return; }
            scale=max(scale,abs(value));
            lower[i]=value;
        }
        if scale==0.0 { atomicOr(&status,2u); return; }
        for (var i=0u;i<n;i++) {
            for (var j=0u;j<i;j++) {
                if abs(lower[i*n+j]-lower[j*n+i])>1e-6*scale { atomicOr(&status,2u); return; }
            }
        }
        for (var i=0u;i<n;i++) {
            for (var j=0u;j<=i;j++) {
                var value=lower[i*n+j];
                for (var k=0u;k<j;k++) { value-=lower[i*n+k]*lower[j*n+k]; }
                if j==i {
                    if !finite(value) || value<=1e-7*scale { atomicOr(&status,2u); return; }
                    lower[i*n+i]=sqrt(value);
                } else { lower[i*n+j]=value/lower[j*n+j]; }
            }
        }
    }
    if empty {
        for (var i=0u;i<n;i++) { if !finite(free_rates[i]) { atomicOr(&status,4u); return; } }
        for (var i=0u;i<n;i++) { rates[i]=free_rates[i]; }
        return;
    }
    for (var i=0u;i<n;i++) {
        var value=rhs[i];
        for (var j=0u;j<i;j++) { value-=lower[i*n+j]*answer[j]; }
        answer[i]=value/lower[i*n+i];
    }
    for (var k=0u;k<n;k++) {
        let i=n-1u-k; var value=answer[i];
        for (var j=i+1u;j<n;j++) { value-=lower[j*n+i]*answer[j]; }
        answer[i]=value/lower[i*n+i];
        if !finite(answer[i]) { atomicOr(&status,4u); return; }
    }
    for (var i=0u;i<n;i++) {
        var value=0.0; var magnitude=abs(rhs[i]);
        for (var j=0u;j<n;j++) { let term=system[i*n+j]*answer[j]; value+=term; magnitude+=abs(term); }
        if !finite(value) || !finite(magnitude) || abs(value-rhs[i])>2e-5*max(magnitude,1e-30) {
            atomicOr(&status,8u); return;
        }
    }
    for (var i=0u;i<n;i++) { rates[i]=answer[i]; }
}

@compute @workgroup_size(64)
fn publish(@builtin(global_invocation_id) id: vec3<u32>) {
    let b=id.x;
    if b>=params.bodies || atomicLoad(&status)!=0u { return; }
    var v=vec3<f32>(0.0); var w=vec3<f32>(0.0);
    for (var i=0u;i<params.width;i++) { v+=column(b,0u,i)*rates[i]; w+=column(b,3u,i)*rates[i]; }
    let m=moments[b];
    let p=linear_momentum(m)-(m.a.x*v-cross(m.a.yzw,w))+boundary_impulse[2u*b].xyz;
    let h=m.d.yzw-(cross(m.a.yzw,v)+inertia(m,w))+boundary_impulse[2u*b+1u].xyz;
    if !finite3(v) || !finite3(w) || !finite3(p) || !finite3(h) { atomicOr(&status,4u); return; }
    motion[2u*b]=vec4<f32>(v,0.0); motion[2u*b+1u]=vec4<f32>(w,0.0);
    impulse[2u*b]=vec4<f32>(p,0.0); impulse[2u*b+1u]=vec4<f32>(h,0.0);
}
