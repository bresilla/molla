struct Contact { soil_mass:vec4<f32>, normal_gap:vec4<f32>, friction:vec4<f32>, boundary:vec4<u32> }
struct Six { k:vec4<f32>, w:vec4<f32> }
struct State { high:Six, low:Six }
struct Iterate { z:Six, dual:Six, local:Six }
struct Params { width:u32, contacts:u32, iterations:u32, pad:u32 }
struct Report { fault:atomic<u32>, steps:u32, factors:u32, pad0:u32, pad1:u32, pad2:u32, pad3:u32, pad4:u32 }
struct Weights { a:vec3<f32>, b:vec3<f32> }
struct FixedContact { normal_rho:vec4<f32>, free:vec4<f32>, a:vec4<f32>, b:vec4<f32> }
struct Active { count:u32, ids:array<u32> }
@group(0) @binding(0) var<storage,read> free_rates:array<vec2<f32>>;
@group(0) @binding(1) var<storage,read> rows:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> responses:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> contacts:array<Contact>;
@group(0) @binding(4) var<storage,read_write> scratch:array<Iterate>;
@group(0) @binding(5) var<storage,read_write> states:array<State>;
@group(0) @binding(6) var<storage,read_write> report:Report;
@group(0) @binding(7) var<uniform> params:Params;
@group(0) @binding(8) var<storage,read_write> active_list:Active;
@group(0) @binding(9) var<storage,read_write> fixed_contacts:array<FixedContact>;
var<workgroup> core:array<f32,1024>;
var<workgroup> pivots:array<u32,32>;
var<workgroup> q:array<f32,32>;
var<workgroup> failed:u32;
var<workgroup> active_count:u32;
var<workgroup> staged_points:array<vec4<f32>,64>;
var<workgroup> staged_responses:array<vec4<f32>,2048>;
fn staged_rates_wide(lane:u32,selected:u32,local:bool) {
    let n=params.width;
    var value=0.0;
    if lane<n && !local {value=free_rates[lane].x+free_rates[lane].y;}
    for(var first=0u;first<selected;first+=64u) {
        let slot=first+lane;
        staged_points[lane]=vec4<f32>(0.0);
        for(var axis=0u;axis<n;axis++) {staged_responses[axis*64u+lane]=vec4<f32>(0.0);}
        if slot<selected {
            let c=active_contact(slot);
            if local {staged_points[lane]=scratch[c].local.k;}else {staged_points[lane]=scratch[c].z.k;}
            for(var axis=0u;axis<n;axis++) {staged_responses[axis*64u+lane]=responses[c*n+axis];}
        }
        workgroupBarrier();
        if lane<n {
            for(var offset=0u;offset<min(64u,selected-first);offset++) {
                if local {value+=dot(staged_responses[lane*64u+offset].xyz,staged_points[offset].xyz);}
                else {value-=dot(staged_responses[lane*64u+offset].xyz,staged_points[offset].xyz);}
            }
        }
        workgroupBarrier();
    }
    if lane<n {q[lane]=value;}
}
fn staged_rates(lane:u32,selected:u32,local:bool) {
    let n=params.width;
    for(var base=0u;base<n;base+=4u) {
        var value=0.0;
        if lane<4u && base+lane<n && !local {value=free_rates[base+lane].x+free_rates[base+lane].y;}
        for(var first=0u;first<selected;first+=64u) {
            let slot=first+lane;
            staged_points[lane]=vec4<f32>(0.0);
            for(var axis=0u;axis<4u;axis++) {staged_responses[axis*64u+lane]=vec4<f32>(0.0);}
            if slot<selected {
                let c=active_contact(slot);
                if local {staged_points[lane]=scratch[c].local.k;}else {staged_points[lane]=scratch[c].z.k;}
                for(var axis=0u;axis<4u;axis++) {if base+axis<n {staged_responses[axis*64u+lane]=responses[c*n+base+axis];}}
            }
            workgroupBarrier();
            if lane<4u && base+lane<n {
                for(var offset=0u;offset<min(64u,selected-first);offset++) {
                    if local {value+=dot(staged_responses[lane*64u+offset].xyz,staged_points[offset].xyz);}
                    else {value-=dot(staged_responses[lane*64u+offset].xyz,staged_points[offset].xyz);}
                }
            }
            workgroupBarrier();
        }
        if lane<4u && base+lane<n {q[base+lane]=value;}
    }
}
fn active_contact(slot:u32)->u32 {
    if params.pad==0u {return slot;}
    return active_list.ids[slot];
}
fn finite(x:f32)->bool {return abs(x)<=3.402823e38;}
fn wall(c:u32,axis:u32)->f32 {
    let flag=(contacts[c].boundary.x>>(2u*axis))&3u;
    return select(select(0.0,1.0,flag==1u),-1.0,flag==2u);
}
fn uncached_weights(c:u32)->Weights {
    let m=contacts[c].soil_mass.w;
    var result:Weights;
    for(var axis=0u;axis<3u;axis++) {
        result.a[axis]=select(0.5*m,(2.0/3.0)*m,wall(c,axis)!=0.0);
        result.b[axis]=select(0.0,-m/3.0,wall(c,axis)!=0.0);
    }
    return result;
}
fn weights(c:u32)->Weights {
    if params.pad==4u {return Weights(fixed_contacts[c].a.xyz,fixed_contacts[c].b.xyz);}
    return uncached_weights(c);
}
fn contact_normal(c:u32)->vec3<f32> {
    if params.pad==4u {return fixed_contacts[c].normal_rho.xyz;}
    return normalize(contacts[c].normal_gap.xyz);
}
fn contact_rho(c:u32)->f32 {
    if params.pad==4u {return fixed_contacts[c].normal_rho.w;}
    return 1.0/contacts[c].soil_mass.w;
}
fn uncached_free(c:u32,normal:vec3<f32>)->vec3<f32> {
    var result=contacts[c].soil_mass.xyz+normal*contacts[c].normal_gap.w;
    for(var j=0u;j<params.width;j++) {result-=rows[c*params.width+j].xyz*(free_rates[j].x+free_rates[j].y);}
    return result;
}
fn contact_free(c:u32,normal:vec3<f32>)->vec3<f32> {
    if params.pad==4u {return fixed_contacts[c].free.xyz;}
    return uncached_free(c,normal);
}
fn cone(n:vec3<f32>,mu:f32,v:vec3<f32>)->vec3<f32> {
    let s=dot(n,v);let tangent=v-n*s;let t=length(tangent);
    if mu==0.0 {return n*max(s,0.0);}
    if s>=0.0 && t<=mu*s {return v;}
    if s+mu*t<=0.0 {return vec3<f32>(0.0);}
    let height=(s+mu*t)/(1.0+mu*mu);
    return n*height+tangent*(mu*height/t);
}
fn factor() {
    let n=params.width;
    for(var k=0u;k<n;k++) {
        var pivot=k;var largest=abs(core[k*32u+k]);
        for(var i=k+1u;i<n;i++) {if abs(core[i*32u+k])>largest {pivot=i;largest=abs(core[i*32u+k]);}}
        if !finite(largest) || largest<=1e-20 {failed=2u;return;}
        pivots[k]=pivot;
        for(var j=0u;j<n;j++) {let value=core[k*32u+j];core[k*32u+j]=core[pivot*32u+j];core[pivot*32u+j]=value;}
        for(var i=k+1u;i<n;i++) {
            core[i*32u+k]/=core[k*32u+k];
            for(var j=k+1u;j<n;j++) {core[i*32u+j]-=core[i*32u+k]*core[k*32u+j];}
        }
    }
    for(var i=0u;i<n;i++) {for(var j=0u;j<n;j++) {if !finite(core[i*32u+j]) {failed=2u;}}}
}
fn solve() {
    let n=params.width;
    for(var k=0u;k<n;k++) {let value=q[k];q[k]=q[pivots[k]];q[pivots[k]]=value;}
    for(var i=0u;i<n;i++) {for(var j=0u;j<i;j++) {q[i]-=core[i*32u+j]*q[j];}}
    for(var reverse=0u;reverse<n;reverse++) {
        let i=n-1u-reverse;
        for(var j=i+1u;j<n;j++) {q[i]-=core[i*32u+j]*q[j];}
        q[i]/=core[i*32u+i];
    }
}
@compute @workgroup_size(64)
fn admm(@builtin(local_invocation_index) lane:u32) {
    let n=params.width;
    if lane==0u {atomicStore(&report.fault,0u);report.steps=0u;report.factors=0u;report.pad0=0u;report.pad1=0u;report.pad2=0u;report.pad3=0u;report.pad4=0u;failed=0u;}
    storageBarrier();
    for(var c=lane;c<max(params.contacts,1u);c+=64u) {
        scratch[c]=Iterate();states[c]=State();fixed_contacts[c]=FixedContact();
        if c>=params.contacts {continue;}
        let node=contacts[c];let m=node.soil_mass.w;
        if m==0.0 {continue;}
        if !all(abs(node.soil_mass)<=vec4<f32>(3.402823e38)) || m<0.0 || !finite(1.0/m)
            || !all(abs(node.normal_gap)<=vec4<f32>(3.402823e38)) || abs(dot(node.normal_gap.xyz,node.normal_gap.xyz)-1.0)>1e-4 || node.normal_gap.w<0.0
            || !finite(node.friction.x) || node.friction.x<0.0 || (node.boundary.x&0xffffffc0u)!=0u {atomicOr(&report.fault,1u);}
        for(var axis=0u;axis<3u;axis++) {if ((node.boundary.x>>(2u*axis))&3u)==3u {atomicOr(&report.fault,1u);}}
        for(var j=0u;j<n;j++) {
            if !all(abs(rows[c*n+j])<=vec4<f32>(3.402823e38)) || !all(abs(responses[c*n+j])<=vec4<f32>(3.402823e38)) {atomicOr(&report.fault,1u);}
        }
    }
    if lane<n {if !all(abs(free_rates[lane])<=vec2<f32>(3.402823e38)) {atomicOr(&report.fault,1u);}}
    storageBarrier();
    if lane==0u {failed=atomicLoad(&report.fault);}
    if workgroupUniformLoad(&failed)!=0u {return;}
    if lane==0u {
        active_count=0u;
        if params.pad!=0u {
            for(var c=0u;c<params.contacts;c++) {
                if contacts[c].soil_mass.w>0.0 {active_list.ids[active_count]=c;active_count++;}
            }
        }
        active_list.count=active_count;
    }
    storageBarrier();
    let selected=select(params.contacts,workgroupUniformLoad(&active_count),params.pad!=0u);
    if params.pad==4u {
        for(var slot=lane;slot<selected;slot+=64u) {
            let c=active_contact(slot);let normal=normalize(contacts[c].normal_gap.xyz);let local=uncached_weights(c);
            fixed_contacts[c]=FixedContact(vec4<f32>(normal,1.0/contacts[c].soil_mass.w),vec4<f32>(uncached_free(c,normal),0.0),vec4<f32>(local.a,0.0),vec4<f32>(local.b,0.0));
        }
    }
    storageBarrier();
    for(var index=lane;index<n*n;index+=64u) {
        let i=index/n;let j=index%n;var value=select(0.0,1.0,i==j);
        for(var slot=0u;slot<selected;slot++) {
            let c=active_contact(slot);
            if contacts[c].soil_mass.w>0.0 {value+=dot(responses[c*n+i].xyz,weights(c).a*rows[c*n+j].xyz);}
        }
        core[i*32u+j]=value;
    }
    workgroupBarrier();
    if lane==0u {factor();report.factors=1u;atomicStore(&report.fault,failed);}
    if workgroupUniformLoad(&failed)!=0u {return;}
    for(var iteration=0u;iteration<params.iterations;iteration++) {
        if params.pad>=3u {staged_rates_wide(lane,selected,false);}else if params.pad==2u {staged_rates(lane,selected,false);}else if lane<n {
            var value=free_rates[lane].x+free_rates[lane].y;
            for(var slot=0u;slot<selected;slot++) {let c=active_contact(slot);if contacts[c].soil_mass.w>0.0 {value-=dot(responses[c*n+lane].xyz,scratch[c].z.k.xyz);}}
            q[lane]=value;
        }
        workgroupBarrier();
        for(var slot=lane;slot<selected;slot+=64u) {
            let c=active_contact(slot);
            let node=contacts[c];if node.soil_mass.w==0.0 {continue;}
            let z=scratch[c].z;let dual=scratch[c].dual;let normal=contact_normal(c);let rho=contact_rho(c);
            var relative=node.soil_mass.xyz+rho*(z.k.xyz+z.w.xyz);let free=contact_free(c,normal);
            for(var j=0u;j<n;j++) {relative-=rows[c*n+j].xyz*q[j];}
            let slip=relative-normal*dot(normal,relative);
            let rk=rho*(z.k.xyz-dual.k.xyz)-free-normal*(node.friction.x*length(slip));
            var rw=rho*(z.w.xyz-dual.w.xyz)-node.soil_mass.xyz;
            for(var axis=0u;axis<3u;axis++) {if wall(c,axis)==0.0 {rw[axis]=0.0;}}
            let local=weights(c);
            scratch[c].local=Six(vec4<f32>(local.a*rk+local.b*rw,0.0),vec4<f32>(local.b*rk+local.a*rw,0.0));
        }
        storageBarrier();
        if params.pad>=3u {staged_rates_wide(lane,selected,true);}else if params.pad==2u {staged_rates(lane,selected,true);}else if lane<n {
            var value=0.0;
            for(var slot=0u;slot<selected;slot++) {let c=active_contact(slot);if contacts[c].soil_mass.w>0.0 {value+=dot(responses[c*n+lane].xyz,scratch[c].local.k.xyz);}}
            q[lane]=value;
        }
        workgroupBarrier();
        if lane==0u {solve();}
        workgroupBarrier();
        for(var slot=lane;slot<selected;slot+=64u) {
            let c=active_contact(slot);
            let node=contacts[c];if node.soil_mass.w==0.0 {continue;}
            var point=vec3<f32>(0.0);for(var j=0u;j<n;j++) {point+=rows[c*n+j].xyz*q[j];}
            let local=weights(c);let y=scratch[c].local;let dual=scratch[c].dual;
            let xk=y.k.xyz-local.a*point;let xw=y.w.xyz-local.b*point;
            let zk=cone(contact_normal(c),node.friction.x,xk+dual.k.xyz);
            var zw=vec3<f32>(0.0);
            for(var axis=0u;axis<3u;axis++) {let sign=wall(c,axis);zw[axis]=sign*max(sign*(xw[axis]+dual.w[axis]),0.0);}
            let uk=dual.k.xyz+xk-zk;let uw=dual.w.xyz+xw-zw;
            if !all(abs(zk)<=vec3<f32>(3.402823e38)) || !all(abs(zw)<=vec3<f32>(3.402823e38)) || !all(abs(uk)<=vec3<f32>(3.402823e38)) || !all(abs(uw)<=vec3<f32>(3.402823e38)) {atomicOr(&report.fault,2u);}
            scratch[c].z=Six(vec4<f32>(zk,0.0),vec4<f32>(zw,0.0));
            scratch[c].dual=Six(vec4<f32>(uk,0.0),vec4<f32>(uw,0.0));
        }
        storageBarrier();
    }
    for(var slot=lane;slot<selected;slot+=64u) {let c=active_contact(slot);states[c].high=scratch[c].z;}
    if lane==0u {report.steps=params.iterations;}
}
