struct Contact { soil_mass:vec4<f32>, normal_gap:vec4<f32>, friction:vec4<f32>, boundary:vec4<u32> }
struct Six { k:vec4<f32>, w:vec4<f32> }
struct State { high:Six, low:Six }
struct Block { residual:array<f32,6>, direct:array<f32,36>, machine:array<f32,18> }
struct Products { forward:Six, transpose:Six }
struct Work { fault:atomic<u32>, pad0:u32, pad1:u32, pad2:u32, q:array<vec2<f32>,32>, coupling:array<vec2<f32>,32> }
struct Params { width:u32, contacts:u32, pad0:u32, pad1:u32 }
@group(0) @binding(0) var<storage,read> free_rates:array<vec2<f32>>;
@group(0) @binding(1) var<storage,read> rows:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> response:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> contacts:array<Contact>;
@group(0) @binding(4) var<storage,read> states:array<State>;
@group(0) @binding(5) var<storage,read> vectors:array<Six>;
@group(0) @binding(6) var<storage,read_write> blocks:array<Block>;
@group(0) @binding(7) var<storage,read_write> products:array<Products>;
@group(0) @binding(8) var<storage,read_write> work:Work;
@group(0) @binding(9) var<uniform> params:Params;
fn finite(x:f32)->bool { return abs(x)<=3.402823e38; }
fn finite4(x:vec4<f32>)->bool { return all(abs(x)<=vec4<f32>(3.402823e38)); }
fn value(x:vec2<f32>)->f32 { return rate_round(x.x+x.y); }
fn six(k:vec3<f32>,w:vec3<f32>)->array<f32,6> { return array<f32,6>(k.x,k.y,k.z,w.x,w.y,w.z); }
fn component(x:Six,i:u32)->f32 { if i<3u { return x.k[i]; } return x.w[i-3u]; }
fn packed(x:array<f32,6>)->Six { return Six(vec4<f32>(x[0],x[1],x[2],0.0),vec4<f32>(x[3],x[4],x[5],0.0)); }
fn flags(c:u32)->vec3<f32> {
    var result=vec3<f32>(0.0);
    for (var i=0u;i<3u;i++) { let flag=(contacts[c].boundary.x>>(2u*i))&3u; if flag!=0u { result[i]=select(1.0,-1.0,flag==2u); } }
    return result;
}
fn valid_six(x:Six)->bool { return finite4(x.k) && finite4(x.w) && x.k.w==0.0 && x.w.w==0.0; }

@compute @workgroup_size(64)
fn state_rates(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x; rate_lane=i%64u;
    if i>=params.width || atomicLoad(&work.fault)!=0u { return; }
    var sum=free_rates[i];
    if !all(abs(sum)<=vec2<f32>(3.402823e38)) { atomicOr(&work.fault,1u); return; }
    for (var c=0u;c<params.contacts;c++) {
        let mass=contacts[c].soil_mass.w;
        if !finite(mass) || mass<0.0 { atomicOr(&work.fault,1u); return; }
        if mass==0.0 { continue; }
        let r=response[c*params.width+i]; let row=rows[c*params.width+i];
        if !finite4(r) || r.w!=0.0 || !finite4(row) || row.w!=0.0 { atomicOr(&work.fault,1u); return; }
        let x=states[c];
        if !valid_six(x.high) || !valid_six(x.low) { atomicOr(&work.fault,1u); return; }
        for (var axis=0u;axis<3u;axis++) { sum=rate_add(sum,rate_mul(-r[axis],vec2<f32>(x.high.k[axis],x.low.k[axis]))); }
    }
    if !all(abs(sum)<=vec2<f32>(3.402823e38)) { atomicOr(&work.fault,2u); return; }
    work.q[i]=sum;
}

struct Local { normal:vec3<f32>, rho:f32, mass:f32, mu:f32, gap:f32, lambda:f32, wall:vec3<f32>, k:vec3<f32>, w:vec3<f32>, soil:vec3<f32>, relative:vec3<f32> }
fn residual(a:Local)->array<f32,6> {
    let t=a.k-a.normal*a.lambda; let slip=a.relative-a.normal*dot(a.normal,a.relative);
    let y=t-a.rho*slip; let radius=length(y); let cap=a.mu*max(a.lambda,0.0);
    let separation=a.rho*(dot(a.normal,a.relative)+a.gap);
    let normal=select(a.lambda,separation,a.lambda-separation>0.0);
    var tangent=a.rho*slip; if radius>cap { tangent=t-y*(cap/radius); }
    let k=a.normal*normal+tangent;
    var w=a.w;
    for (var i=0u;i<3u;i++) { if a.wall[i]*(a.w[i]-a.mass*a.soil[i])>0.0 { w[i]=a.mass*a.soil[i]; } }
    return six(k,w);
}
fn derivative(a:Local,dk:vec3<f32>,dw:vec3<f32>,ds:vec3<f32>,dr:vec3<f32>)->array<f32,6> {
    let dl=dot(a.normal,dk); let t=a.k-a.normal*a.lambda; let dt=dk-a.normal*dl;
    let slip=a.relative-a.normal*dot(a.normal,a.relative); let dslip=dr-a.normal*dot(a.normal,dr);
    let y=t-a.rho*slip; let dy=dt-a.rho*dslip; let radius=length(y); let cap=a.mu*max(a.lambda,0.0);
    var dp=dy;
    if a.mu==0.0 { dp=vec3<f32>(0.0); }
    else if radius>cap {
        let unit=y/radius; let dc=select(0.0,a.mu*dl,a.lambda>0.0);
        dp=unit*dc+(cap/radius)*(dy-unit*dot(unit,dy));
    }
    var dn=dl; if a.lambda-a.rho*(dot(a.normal,a.relative)+a.gap)>0.0 { dn=a.rho*dot(a.normal,dr); }
    let k=a.normal*dn+dt-dp; var w=dw;
    for (var i=0u;i<3u;i++) { if a.wall[i]*(a.w[i]-a.mass*a.soil[i])>0.0 { w[i]=a.mass*ds[i]; } }
    return six(k,w);
}

@compute @workgroup_size(64)
fn linearize(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x; rate_lane=c%64u;
    if c>=max(params.contacts,1u) { return; }
    blocks[c]=Block();
    if c>=params.contacts || atomicLoad(&work.fault)!=0u { return; }
    let node=contacts[c]; if node.soil_mass.w==0.0 { return; }
    let normal_size=dot(node.normal_gap.xyz,node.normal_gap.xyz);
    if !finite4(node.soil_mass) || !finite4(node.normal_gap) || abs(normal_size-1.0)>1e-4 || node.normal_gap.w<0.0
        || !finite4(node.friction) || node.friction.x<0.0 || any(node.friction.yzw!=vec3<f32>(0.0))
        || (node.boundary.x & ~63u)!=0u || any(node.boundary.yzw!=vec3<u32>(0u)) { atomicOr(&work.fault,1u); return; }
    for (var i=0u;i<3u;i++) { if ((node.boundary.x>>(2u*i))&3u)==3u { atomicOr(&work.fault,1u); return; } }
    let x=states[c]; var a:Local; a.normal=normalize(node.normal_gap.xyz); a.mass=node.soil_mass.w;
    a.mu=node.friction.x; a.gap=node.normal_gap.w; a.wall=flags(c);
    var trace=vec2<f32>(0.0);
    for (var axis=0u;axis<3u;axis++) {
        a.k[axis]=value(vec2<f32>(x.high.k[axis],x.low.k[axis]));
        a.w[axis]=value(vec2<f32>(x.high.w[axis],x.low.w[axis]));
        let impulse=rate_add(vec2<f32>(x.high.k[axis],x.low.k[axis]),vec2<f32>(x.high.w[axis],x.low.w[axis]));
        let soil=rate_add(rate_mul(a.mass,vec2<f32>(node.soil_mass[axis],0.0)),impulse);
        let soil_velocity=rate_mul(1.0/a.mass,soil); var machine=vec2<f32>(0.0);
        for (var i=0u;i<params.width;i++) {
            machine=rate_add(machine,rate_mul(rows[c*params.width+i][axis],work.q[i]));
            trace=rate_add(trace,rate_mul(rows[c*params.width+i][axis],vec2<f32>(response[c*params.width+i][axis],0.0)));
        }
        a.soil[axis]=value(soil_velocity); a.relative[axis]=value(rate_add(soil_velocity,-machine));
    }
    a.lambda=dot(a.normal,a.k); let scale=1.0/a.mass+value(trace);
    if !finite(scale) || scale<=0.0 { atomicOr(&work.fault,2u); return; }
    a.rho=1.0/scale; var block:Block; block.residual=residual(a);
    for (var j=0u;j<6u;j++) {
        var dk=vec3<f32>(0.0); var dw=vec3<f32>(0.0);
        if j<3u { dk[j]=1.0; } else { dw[j-3u]=1.0; }
        let ds=(dk+dw)/a.mass; let column=derivative(a,dk,dw,ds,ds);
        for (var i=0u;i<6u;i++) { block.direct[6u*i+j]=column[i]; }
    }
    for (var j=0u;j<3u;j++) {
        var dr=vec3<f32>(0.0); dr[j]=-1.0;
        let column=derivative(a,vec3<f32>(0.0),vec3<f32>(0.0),vec3<f32>(0.0),dr);
        for (var i=0u;i<6u;i++) { block.machine[3u*i+j]=column[i]; }
    }
    for (var i=0u;i<6u;i++) { if !finite(block.residual[i]) { atomicOr(&work.fault,2u); return; } }
    for (var i=0u;i<36u;i++) { if !finite(block.direct[i]) { atomicOr(&work.fault,2u); return; } }
    for (var i=0u;i<18u;i++) { if !finite(block.machine[i]) { atomicOr(&work.fault,2u); return; } }
    blocks[c]=block;
}

@compute @workgroup_size(64)
fn vector_rates(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x; rate_lane=i%64u;
    if i>=params.width || atomicLoad(&work.fault)!=0u { return; }
    var sum=vec2<f32>(0.0);
    for (var c=0u;c<params.contacts;c++) {
        if contacts[c].soil_mass.w==0.0 { continue; }
        let v=vectors[c]; if !valid_six(v) { atomicOr(&work.fault,1u); return; }
        for (var axis=0u;axis<3u;axis++) { sum=rate_add(sum,rate_mul(-response[c*params.width+i][axis],vec2<f32>(v.k[axis],0.0))); }
    }
    if !all(abs(sum)<=vec2<f32>(3.402823e38)) { atomicOr(&work.fault,2u); return; }
    work.coupling[i]=sum;
}

@compute @workgroup_size(64)
fn forward_product(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x; rate_lane=c%64u;
    if c>=max(params.contacts,1u) { return; }
    products[c].forward=Six();
    if c>=params.contacts || atomicLoad(&work.fault)!=0u || contacts[c].soil_mass.w==0.0 { return; }
    var point:array<vec2<f32>,3>;
    for (var axis=0u;axis<3u;axis++) { for (var i=0u;i<params.width;i++) { point[axis]=rate_add(point[axis],rate_mul(rows[c*params.width+i][axis],work.coupling[i])); } }
    var result:array<f32,6>;
    for (var i=0u;i<6u;i++) {
        var sum=vec2<f32>(0.0);
        for (var j=0u;j<6u;j++) { sum=rate_add(sum,rate_mul(blocks[c].direct[6u*i+j],vec2<f32>(component(vectors[c],j),0.0))); }
        for (var j=0u;j<3u;j++) { sum=rate_add(sum,rate_mul(blocks[c].machine[3u*i+j],point[j])); }
        result[i]=value(sum); if !finite(result[i]) { atomicOr(&work.fault,2u); return; }
    }
    products[c].forward=packed(result);
}

@compute @workgroup_size(64)
fn transpose_rates(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x; rate_lane=i%64u;
    if i>=params.width || atomicLoad(&work.fault)!=0u { return; }
    var sum=vec2<f32>(0.0);
    for (var c=0u;c<params.contacts;c++) {
        if contacts[c].soil_mass.w==0.0 { continue; }
        if !valid_six(vectors[c]) { atomicOr(&work.fault,1u); return; }
        for (var axis=0u;axis<3u;axis++) {
            var point=vec2<f32>(0.0);
            for (var j=0u;j<6u;j++) { point=rate_add(point,rate_mul(blocks[c].machine[3u*j+axis],vec2<f32>(component(vectors[c],j),0.0))); }
            sum=rate_add(sum,rate_mul(rows[c*params.width+i][axis],point));
        }
    }
    if !all(abs(sum)<=vec2<f32>(3.402823e38)) { atomicOr(&work.fault,2u); return; }
    work.coupling[i]=sum;
}

@compute @workgroup_size(64)
fn transpose_product(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x; rate_lane=c%64u;
    if c>=max(params.contacts,1u) { return; }
    products[c].transpose=Six();
    if c>=params.contacts || atomicLoad(&work.fault)!=0u || contacts[c].soil_mass.w==0.0 { return; }
    var result:array<f32,6>;
    for (var j=0u;j<6u;j++) {
        var sum=vec2<f32>(0.0);
        for (var i=0u;i<6u;i++) { sum=rate_add(sum,rate_mul(blocks[c].direct[6u*i+j],vec2<f32>(component(vectors[c],i),0.0))); }
        if j<3u { for (var i=0u;i<params.width;i++) { sum=rate_add(sum,rate_mul(-response[c*params.width+i][j],work.coupling[i])); } }
        result[j]=value(sum); if !finite(result[j]) { atomicOr(&work.fault,2u); return; }
    }
    products[c].transpose=packed(result);
}
