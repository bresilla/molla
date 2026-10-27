struct Contact { soil_mass:vec4<f32>, normal_gap:vec4<f32>, friction:vec4<f32>, boundary:vec4<u32> }
struct Six { k:vec4<f32>, w:vec4<f32> }
struct State { high:Six, low:Six }
struct Output { machine_impulse_normal:vec4<f32>, soil_velocity_energy:vec4<f32>, machine_velocity_residual:vec4<f32>, wall_impulse:vec4<f32>, status:vec4<u32> }
struct Status { fault:atomic<u32>, contact:atomic<u32>, active_count:atomic<u32>, solver_fault:u32, energy:f32, scale:f32, pad0:u32, pad1:u32 }
struct Params { width:u32, contacts:u32, pad0:u32, pad1:u32 }
@group(0) @binding(0) var<storage,read> mass:array<f32>;
@group(0) @binding(1) var<storage,read> free_rates:array<vec2<f32>>;
@group(0) @binding(2) var<storage,read> rows:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> response:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> contacts:array<Contact>;
@group(0) @binding(5) var<storage,read> states:array<State>;
@group(0) @binding(6) var<storage,read> upstream:array<u32>;
@group(0) @binding(7) var<storage,read> solver_status:array<u32>;
@group(0) @binding(8) var<storage,read_write> outputs:array<Output>;
@group(0) @binding(9) var<storage,read_write> rates:array<vec2<f32>>;
@group(0) @binding(10) var<storage,read_write> status:Status;
@group(0) @binding(11) var<uniform> params:Params;
var<workgroup> q:array<vec2<f32>,32>;
var<workgroup> energy_parts:array<vec2<f32>,64>;
var<workgroup> scales:array<f32,64>;
var<workgroup> stop:u32;
fn finite(x:f32)->bool {return abs(x)<=3.402823e38;}
fn finite4(x:vec4<f32>)->bool {return all(abs(x)<=vec4<f32>(3.402823e38));}
fn value(x:vec2<f32>)->f32 {return rate_round(x.x+x.y);}
fn product(a:vec2<f32>,b:vec2<f32>)->vec2<f32> {return rate_add(rate_mul(a.x,b),rate_mul(a.y,b));}
fn publication_tangent(v:array<vec2<f32>,3>,n:vec3<f32>)->vec3<f32> {
    var crossed:array<vec2<f32>,3>;var norm=vec2<f32>(0.0);
    for(var i=0u;i<3u;i++) {
        let j=(i+1u)%3u;let k=(i+2u)%3u;
        crossed[i]=rate_add(rate_mul(n[k],v[j]),-rate_mul(n[j],v[k]));
        norm=rate_add(norm,rate_mul(n[i],vec2<f32>(n[i],0.0)));
    }
    var tangent=vec3<f32>(0.0);
    for(var i=0u;i<3u;i++) {
        let j=(i+1u)%3u;let k=(i+2u)%3u;
        tangent[i]=value(rate_add(rate_mul(n[j],crossed[k]),-rate_mul(n[k],crossed[j])))/value(norm);
    }
    return tangent;
}
fn fault(bits:u32,c:u32) {atomicOr(&status.fault,bits);if c>0u {atomicMin(&status.contact,c);}}
fn valid_six(x:Six)->bool {return finite4(x.k) && finite4(x.w) && x.k.w==0.0 && x.w.w==0.0;}
@compute @workgroup_size(64)
fn certify(@builtin(local_invocation_index) lane:u32) {
    rate_lane=lane;
    if lane==0u {
        atomicStore(&status.contact,0xffffffffu);atomicStore(&status.active_count,0u);
        status.solver_fault=solver_status[0];status.energy=0.0;status.scale=1.0;status.pad0=0u;status.pad1=0u;
        if upstream[0]!=0u || (solver_status[0]&~24u)!=0u {fault(1u,0u);}
    }
    storageBarrier();
    for (var c=lane;c<max(params.contacts,1u);c+=64u) {
        outputs[c]=Output();if c>=params.contacts {continue;}
        let node=contacts[c];
        if !finite(node.soil_mass.w) || node.soil_mass.w<0.0 {fault(1u,c+1u);continue;}
        if node.soil_mass.w==0.0 {continue;}
        atomicAdd(&status.active_count,1u);
        let x=states[c];
        if !finite4(node.soil_mass) || !finite4(node.normal_gap) || abs(dot(node.normal_gap.xyz,node.normal_gap.xyz)-1.0)>1e-4 || node.normal_gap.w<0.0
            || !finite4(node.friction) || node.friction.x<0.0 || any(node.friction.yzw!=vec3<f32>(0.0)) || (node.boundary.x&~63u)!=0u || any(node.boundary.yzw!=vec3<u32>(0u)) || !valid_six(x.high) || !valid_six(x.low) {fault(1u,c+1u);continue;}
        var k=vec3<f32>(0.0);var w=vec3<f32>(0.0);
        for (var axis=0u;axis<3u;axis++) {
            if ((node.boundary.x>>(2u*axis))&3u)==3u {fault(1u,c+1u);}
            k[axis]=value(vec2<f32>(x.high.k[axis],x.low.k[axis]));w[axis]=value(vec2<f32>(x.high.w[axis],x.low.w[axis]));
        }
        for (var i=0u;i<params.width;i++) {
            let a=rows[c*params.width+i];let b=response[c*params.width+i];
            if !finite4(a) || !finite4(b) || a.w!=0.0 || b.w!=0.0 {fault(1u,c+1u);}
        }
        if !all(abs(k)<=vec3<f32>(3.402823e38)) || !all(abs(w)<=vec3<f32>(3.402823e38)) {fault(2u,c+1u);}
        outputs[c].machine_impulse_normal=vec4<f32>(-k,dot(normalize(node.normal_gap.xyz),k));outputs[c].wall_impulse=vec4<f32>(w,0.0);
    }
    if lane<params.width {
        rates[lane]=vec2<f32>(0.0);q[lane]=vec2<f32>(0.0);
        if !all(abs(free_rates[lane])<=vec2<f32>(3.402823e38)) || !finite(value(free_rates[lane])) {fault(1u,0u);}
        for (var j=0u;j<params.width;j++) {
            let m=mass[lane*params.width+j];
            if !finite(m) || (lane==j && m<=0.0) || abs(m-mass[j*params.width+lane])>1e-6*max(1.0,abs(m)) {fault(1u,0u);}
        }
    }
    storageBarrier();
    if lane==0u {stop=atomicLoad(&status.fault);}
    let invalid_inputs=workgroupUniformLoad(&stop)!=0u;
    if lane<params.width && !invalid_inputs {
        var sum=free_rates[lane];
        for (var c=0u;c<params.contacts;c++) {if contacts[c].soil_mass.w>0.0 {
            let k=-outputs[c].machine_impulse_normal.xyz;
            for (var axis=0u;axis<3u;axis++) {sum=rate_add(sum,rate_mul(-response[c*params.width+lane][axis],vec2<f32>(k[axis],0.0)));}
        }}
        q[lane]=sum;if !all(abs(sum)<=vec2<f32>(3.402823e38)) || !finite(value(sum)) {fault(2u,0u);}
    }
    workgroupBarrier();storageBarrier();
    if lane==0u {stop=atomicLoad(&status.fault);}
    let invalid_rates=workgroupUniformLoad(&stop)!=0u;
    var energy=vec2<f32>(0.0);var scale=0.0;
    for (var c=lane;c<params.contacts;c+=64u) {
        let node=contacts[c];if node.soil_mass.w==0.0 || invalid_rates {continue;}
        let k=-outputs[c].machine_impulse_normal.xyz;var w=outputs[c].wall_impulse.xyz;let normal=normalize(node.normal_gap.xyz);
        var soil=vec3<f32>(0.0);var machine=vec3<f32>(0.0);var relative=vec3<f32>(0.0);
        var machine_pairs:array<vec2<f32>,3>;
        var relative_pairs:array<vec2<f32>,3>;
        for (var axis=0u;axis<3u;axis++) {
            let state=states[c];
            let impulse=rate_add(vec2<f32>(state.high.k[axis],state.low.k[axis]),vec2<f32>(state.high.w[axis],state.low.w[axis]));
            soil[axis]=value(rate_mul(1.0/node.soil_mass.w,rate_add(rate_mul(node.soil_mass.w,vec2<f32>(node.soil_mass[axis],0.0)),impulse)));
            var speed=vec2<f32>(0.0);
            for (var i=0u;i<params.width;i++) {speed=rate_add(speed,rate_mul(rows[c*params.width+i][axis],q[i]));}
            machine[axis]=value(speed);machine_pairs[axis]=speed;
        }
        let raw_vs=1.0+length(soil)+length(machine);let raw_ps=1.0+length(k)+length(w);
        if !finite(raw_vs) || !finite(raw_ps) {fault(2u,c+1u);continue;}
        for (var axis=0u;axis<3u;axis++) {
            let flag=(node.boundary.x>>(2u*axis))&3u;let sign=select(1.0,-1.0,flag==2u);
            if flag!=0u && sign*soil[axis]<0.0 {
                if sign*soil[axis]< -1e-6*raw_vs || sign*w[axis]< -1e-6*raw_ps {fault(16u,c+1u);continue;}
                let reaction=value(rate_add(rate_mul(-node.soil_mass.w,vec2<f32>(node.soil_mass[axis],0.0)),vec2<f32>(-k[axis],0.0)));
                if !finite(reaction) {fault(2u,c+1u);continue;}
                w[axis]=sign*max(0.0,sign*reaction);soil[axis]=0.0;
            }
            relative_pairs[axis]=rate_add(vec2<f32>(soil[axis],0.0),-machine_pairs[axis]);
            relative[axis]=value(relative_pairs[axis]);
        }
        var normal_machine=vec2<f32>(0.0);
        for (var i=0u;i<params.width;i++) {normal_machine=normal_rate_add(normal_machine,node.normal_gap.xyz,rows[c*params.width+i].xyz,q[i]);}
        let strict=normal_rate_residual(soil,node.normal_gap.xyz,node.normal_gap.w,normal_machine);
        let lambda=dot(k,normal);let tangent=publication_tangent(array<vec2<f32>,3>(vec2<f32>(k.x,0.0),vec2<f32>(k.y,0.0),vec2<f32>(k.z,0.0)),node.normal_gap.xyz);let cap=node.friction.x*lambda;
        let separation=dot(normal,relative)+node.normal_gap.w;let slip=publication_tangent(relative_pairs,node.normal_gap.xyz);
        let vs=1.0+length(soil)+length(machine);let ps=1.0+length(k)+length(w);
        if !finite(vs) || !finite(ps) || !finite(cap) || !finite(lambda) || !finite(separation) || !all(abs(strict)<=vec2<f32>(3.402823e38)) || !all(abs(relative)<=vec3<f32>(3.402823e38)) {fault(2u,c+1u);continue;}
        var momentum=vec3<f32>(0.0);var local_energy=vec2<f32>(0.0);
        for (var axis=0u;axis<3u;axis++) {
            let delta=rate_sum(soil[axis],-node.soil_mass[axis]);
            momentum[axis]=value(rate_add(rate_mul(node.soil_mass.w,delta),-rate_sum(k[axis],w[axis])));
            local_energy=rate_add(local_energy,rate_mul(0.5*node.soil_mass.w,product(delta,rate_sum(soil[axis],node.soil_mass[axis]))));
        }
        let momentum_scale=ps+length(node.soil_mass.w*node.soil_mass.xyz);
        if !finite(momentum_scale) || !all(abs(momentum)<=vec3<f32>(3.402823e38)) {fault(2u,c+1u);}
        else if length(momentum)>1e-6*momentum_scale {fault(4u,c+1u);}
        if lambda< -1e-6*ps || strict.x< -1e-6*strict.y || abs(lambda*separation)>1e-6*ps*vs || length(tangent)>cap+1e-6*ps
            || (length(slip)>1e-6*vs && length(tangent+cap*normalize(slip))>1e-6*ps) {fault(8u,c+1u);}
        var wall_bits=0u;
        for (var axis=0u;axis<3u;axis++) {
            let flag=(node.boundary.x>>(2u*axis))&3u;
            if flag==0u {if w[axis]!=0.0 {fault(16u,c+1u);}}
            else {let sign=select(1.0,-1.0,flag==2u);
                if sign*w[axis]< -1e-6*ps || sign*soil[axis]< -1e-6*vs || abs(w[axis]*soil[axis])>1e-6*ps*vs {fault(16u,c+1u);}
                if sign*w[axis]>0.0 {wall_bits|=1u<<axis;}
            }
        }
        let e=value(local_energy);energy=rate_add(energy,local_energy);scale+=abs(e);
        outputs[c].wall_impulse=vec4<f32>(w,0.0);outputs[c].soil_velocity_energy=vec4<f32>(soil,e);outputs[c].machine_velocity_residual=vec4<f32>(machine,0.0);outputs[c].status=vec4<u32>(0u,0u,0u,wall_bits);
    }
    energy_parts[lane]=energy;scales[lane]=scale;workgroupBarrier();storageBarrier();
    certify_machine(lane);
    storageBarrier();
    if lane==0u {stop=atomicLoad(&status.fault);if atomicLoad(&status.contact)==0xffffffffu {atomicStore(&status.contact,0u);}}
    let rejected=workgroupUniformLoad(&stop)!=0u;
    for (var c=lane;c<max(params.contacts,1u);c+=64u) {if rejected {outputs[c]=Output();}}
    if lane<params.width {rates[lane]=select(q[lane],vec2<f32>(0.0),rejected);}
}
