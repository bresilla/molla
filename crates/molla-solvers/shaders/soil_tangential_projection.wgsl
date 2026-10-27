struct Params { width:u32, contacts:u32, iterations:u32, pad:u32 }
struct Status { fault:atomic<u32>, iterations:u32, backtracks:u32, pad:u32 }
@group(0) @binding(0) var<storage,read> mass:array<f32>;
@group(0) @binding(1) var<storage,read> free_rates:array<f32>;
@group(0) @binding(2) var<storage,read> rows:array<vec2<f32>>;
@group(0) @binding(3) var<storage,read> contacts:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> rates:array<f32>;
@group(0) @binding(5) var<storage,read_write> outputs:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> derivatives:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> status:Status;
@group(0) @binding(8) var<uniform> params:Params;
var<workgroup> hessian:array<f32,1024>;
var<workgroup> lower:array<f32,1024>;
var<workgroup> rate:array<f32,32>;
var<workgroup> rate_tail:array<f32,32>;
var<workgroup> arithmetic_round:array<atomic<u32>,64>;
var<private> rounding_lane:u32;
var<workgroup> gradient:array<f32,32>;
var<workgroup> direction:array<f32,32>;
var<workgroup> reduction:array<f32,64>;
var<workgroup> running:u32;
var<workgroup> done:u32;
var<workgroup> step:f32;
var<workgroup> slope:f32;
fn finite(x:f32)->bool { return abs(x)<=3.402823e38; }
fn rounded(value:f32)->f32 {
    atomicStore(&arithmetic_round[rounding_lane],bitcast<u32>(value));
    return bitcast<f32>(atomicAdd(&arithmetic_round[rounding_lane],0u));
}
fn two_sum(a:f32,b:f32)->vec2<f32> {
    let sum=rounded(a+b); let virtual_b=rounded(sum-a);
    let error_a=rounded(a-rounded(sum-virtual_b));
    let error_b=rounded(b-virtual_b);
    return vec2<f32>(sum,rounded(error_a+error_b));
}
fn norm(v:vec2<f32>)->f32 {
    let scale=max(abs(v.x),abs(v.y));
    if scale==0.0 { return 0.0; }
    return scale*length(v/scale);
}
fn factor()->bool {
    let n=params.width;
    for (var i=0u;i<n;i++) { for (var j=0u;j<=i;j++) {
        var value=hessian[i*n+j];
        for (var k=0u;k<j;k++) { value-=lower[i*n+k]*lower[j*n+k]; }
        if !finite(value) || (i==j && value<=0.0) { return false; }
        if i==j { lower[i*n+j]=sqrt(value); }
        else { lower[i*n+j]=value/lower[j*n+j]; }
    } }
    return true;
}
fn relative(c:u32)->vec2<f32> {
    var result=vec2<f32>(0.0);
    for (var axis=0u;axis<2u;axis++) {
        var value=vec2<f32>(0.0);
        for (var i=0u;i<params.width;i++) {
            let a=rows[c*params.width+i][axis];
            let p=rounded(a*rate[i]);
            let error=fma(a,rate[i],-p)+a*rate_tail[i];
            let sum=two_sum(value.x,p);
            value=two_sum(sum.x,value.y+sum.y+error);
        }
        let offset=two_sum(value.x,-contacts[c][axis+2u]);
        result[axis]=rounded(offset.x+rounded(value.y+offset.y));
    }
    return result;
}
fn contact(c:u32) {
    outputs[c]=vec4<f32>(0.0,0.0,contacts[c].zw);
    derivatives[c]=vec4<f32>(0.0);
    let m=contacts[c].x; let cap=contacts[c].y;
    if m==0.0 || cap==0.0 { return; }
    let v=relative(c); let r=norm(v);
    if !finite(r) { atomicOr(&status.fault,4u); return; }
    var impulse=m*v;
    var d=vec3<f32>(m,0.0,m);
    if r>cap/m {
        let unit=v/r;
        impulse=cap*unit;
        d=(cap/r)*vec3<f32>(unit.y*unit.y,-unit.x*unit.y,unit.x*unit.x);
    }
    let corrected=contacts[c].zw+impulse/m;
    if !all(abs(impulse)<=vec2<f32>(3.402823e38)) || !all(abs(corrected)<=vec2<f32>(3.402823e38))
        || !all(abs(d)<=vec3<f32>(3.402823e38)) { atomicOr(&status.fault,4u); return; }
    outputs[c]=vec4<f32>(-impulse,corrected);
    derivatives[c]=vec4<f32>(d,0.0);
}
fn assemble(i:u32) {
    let n=params.width;
    var value=vec2<f32>(0.0);
    for (var j=0u;j<n;j++) {
        hessian[i*n+j]=mass[i*n+j];
        let delta=two_sum(rate[j],-free_rates[j]);
        let product=rounded(mass[i*n+j]*delta.x);
        let error=fma(mass[i*n+j],delta.x,-product)+mass[i*n+j]*(delta.y+rate_tail[j]);
        let sum=two_sum(value.x,product);
        value=two_sum(sum.x,value.y+sum.y+error);
    }
    for (var c=0u;c<params.contacts;c++) {
        let a=rows[c*n+i]; let d=derivatives[c].xyz;
        for (var axis=0u;axis<2u;axis++) {
            let p=rounded(-a[axis]*outputs[c][axis]);
            let sum=two_sum(value.x,p);
            value=two_sum(sum.x,value.y+sum.y+fma(-a[axis],outputs[c][axis],-p));
        }
        let weighted=vec2<f32>(dot(a,d.xy),dot(a,d.yz));
        for (var j=0u;j<n;j++) { hessian[i*n+j]+=dot(weighted,rows[c*n+j]); }
    }
    gradient[i]=rounded(value.x+value.y);
}
fn prepare(iteration:u32) {
    status.iterations=iteration+1u;
    let n=params.width;
    if !factor() { atomicOr(&status.fault,2u); running=0u; return; }
    for (var i=0u;i<n;i++) {
        var v=-gradient[i];
        for (var j=0u;j<i;j++) { v-=lower[i*n+j]*direction[j]; }
        direction[i]=v/lower[i*n+i];
    }
    slope=0.0;
    var converged=true;
    for (var k=0u;k<n;k++) {
        let i=n-1u-k; var v=direction[i];
        for (var j=i+1u;j<n;j++) { v-=lower[j*n+i]*direction[j]; }
        direction[i]=v/lower[i*n+i];
        if !finite(direction[i]) || !finite(gradient[i]) { atomicOr(&status.fault,4u); running=0u; return; }
        var scale=1.0;
        for (var j=0u;j<n;j++) { scale+=abs(mass[i*n+j]*free_rates[j]); }
        converged=converged && abs(gradient[i])<=1e-6*scale && abs(direction[i])<=1e-6*max(1.0,abs(rate[i]));
        slope+=gradient[i]*direction[i];
    }
    if converged {
        for (var i=0u;i<n;i++) { rates[i]=rounded(rate[i]+rate_tail[i]); }
        done=1u; running=0u; return;
    }
    if !finite(slope) || slope>=0.0 { atomicOr(&status.fault,8u); running=0u; }
    step=1.0;
}
fn difference(c:u32)->f32 {
    let m=contacts[c].x; let cap=contacts[c].y;
    if m==0.0 || cap==0.0 { return 0.0; }
    let old=relative(c); var delta=vec2<f32>(0.0);
    for (var i=0u;i<params.width;i++) { delta+=rows[c*params.width+i]*(step*direction[i]); }
    let r0=norm(old); let r1=norm(old+delta); let sum=r0+r1;
    if sum==0.0 { return 0.0; }
    let change=dot(delta,2.0*old+delta)/sum;
    let threshold=cap/m;
    if max(r0,r1)<=threshold { return 0.5*m*sum*change; }
    if min(r0,r1)>=threshold { return cap*change; }
    let lo=min(r0,r1); let hi=max(r0,r1);
    let fraction=(threshold-lo)/(hi-lo);
    return change*(fraction*0.5*m*(threshold+lo)+(1.0-fraction)*cap);
}
@compute @workgroup_size(64)
fn solve(@builtin(local_invocation_index) lane:u32) {
    rounding_lane=lane;
    if lane==0u {
        running=u32(atomicLoad(&status.fault)==0u); done=0u;
        status.iterations=0u; status.backtracks=0u; status.pad=0u;
        for (var i=0u;i<params.width;i++) {
            rate[i]=free_rates[i]; rate_tail[i]=0.0;
            if !finite(rate[i]) { atomicOr(&status.fault,1u); }
            for (var j=0u;j<params.width;j++) {
                let v=mass[i*params.width+j]; hessian[i*params.width+j]=v;
                if !finite(v) || v!=mass[j*params.width+i] { atomicOr(&status.fault,2u); }
            }
        }
        if !factor() { atomicOr(&status.fault,2u); }
    }
    if workgroupUniformLoad(&running)==0u { return; }
    for (var c=lane;c<params.contacts;c+=64u) {
        let value=contacts[c];
        if !all(abs(value)<=vec4<f32>(3.402823e38)) || value.x<0.0 || value.y<0.0 || (value.x==0.0 && value.y!=0.0) { atomicOr(&status.fault,1u); }
        for (var i=0u;i<params.width;i++) {
            if !all(abs(rows[c*params.width+i])<=vec2<f32>(3.402823e38)) { atomicOr(&status.fault,1u); }
        }
    }
    storageBarrier();
    for (var iteration=0u;iteration<params.iterations;iteration++) {
        for (var c=lane;c<params.contacts;c+=64u) { contact(c); }
        storageBarrier();
        if lane<params.width { assemble(lane); }
        workgroupBarrier();
        if lane==0u {
            if atomicLoad(&status.fault)==0u { prepare(iteration); } else { running=0u; }
        }
        if workgroupUniformLoad(&running)==0u { return; }
        for (var backtrack=0u;backtrack<32u;backtrack++) {
            var change=0.0;
            for (var c=lane;c<params.contacts;c+=64u) { change+=difference(c); }
            if lane<params.width {
                let i=lane;
                for (var j=0u;j<params.width;j++) {
                    let delta=two_sum(rate[j],-free_rates[j]);
                    change+=step*direction[i]*mass[i*params.width+j]*(delta.x+(delta.y+rate_tail[j])+0.5*step*direction[j]);
                }
            }
            reduction[lane]=change;
            workgroupBarrier();
            if lane==0u {
                var energy=0.0;
                for (var i=0u;i<64u;i++) { energy+=reduction[i]; }
                if finite(energy) && energy<=1e-4*step*slope { done=1u; }
                else { step*=0.5; status.backtracks+=1u; }
            }
            if workgroupUniformLoad(&done)!=0u { break; }
        }
        if lane==0u {
            if done==0u { atomicOr(&status.fault,8u); running=0u; }
            else {
                for (var i=0u;i<params.width;i++) {
                    let delta=two_sum(rate[i],step*direction[i]);
                    let next=two_sum(delta.x,delta.y+rate_tail[i]);
                    rate[i]=next.x; rate_tail[i]=next.y;
                }
                done=0u;
            }
        }
        if workgroupUniformLoad(&running)==0u { return; }
        storageBarrier();
    }
    if lane==0u { atomicOr(&status.fault,16u); }
}
