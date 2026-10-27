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
@group(0) @binding(9) var<storage,read> normal_rows:array<f32>;
@group(0) @binding(10) var<storage,read> normal_data:array<vec4<f32>>;
@group(0) @binding(11) var<storage,read_write> normal_outputs:array<vec4<f32>>;
var<workgroup> jacobian:array<f32,1024>;
var<workgroup> lower:array<f32,1024>;
var<workgroup> rate:array<f32,32>;
var<workgroup> rate_tail:array<f32,32>;
var<workgroup> base_rate:array<f32,32>;
var<workgroup> base_tail:array<f32,32>;
var<workgroup> merit:f32;
var<workgroup> arithmetic_round:array<atomic<u32>,64>;
var<private> rounding_lane:u32;
var<workgroup> gradient:array<f32,32>;
var<workgroup> direction:array<f32,32>;
var<workgroup> reduction:array<f32,64>;
var<workgroup> running:u32;
var<workgroup> done:u32;
var<workgroup> step:f32;
var<workgroup> slope:f32;
var<workgroup> active_end:atomic<u32>;
fn contact_count()->u32 { return atomicLoad(&active_end); }
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
        var value=jacobian[i*n+j];
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
fn normal_relative(c:u32)->f32 {
    var value=vec2<f32>(0.0);
    for (var i=0u;i<params.width;i++) {
        let a=normal_rows[c*params.width+i];
        let p=rounded(a*rate[i]);
        let sum=two_sum(value.x,p);
        value=two_sum(sum.x,value.y+sum.y+fma(a,rate[i],-p)+a*rate_tail[i]);
    }
    let velocity=two_sum(value.x,-normal_data[c].x);
    let gap=two_sum(velocity.x,-normal_data[c].y);
    return rounded(gap.x+rounded(value.y+velocity.y+gap.y));
}
fn contact(c:u32) {
    outputs[c]=vec4<f32>(0.0,0.0,contacts[c].zw);
    derivatives[c]=vec4<f32>(0.0);
    let m=contacts[c].x;
    normal_outputs[c].x=0.0; normal_outputs[c].y=normal_data[c].x; normal_outputs[c].w=0.0;
    if m>0.0 {
        let z=normal_relative(c); let impulse=m*max(z,0.0);
        let velocity=normal_data[c].x+impulse/m;
        if !finite(z) || !finite(impulse) || !finite(velocity) { atomicOr(&status.fault,4u); return; }
        normal_outputs[c].x=impulse; normal_outputs[c].y=velocity;
        normal_outputs[c].w=select(0.0,m,z>0.0);
    }
    let cap=contacts[c].y*normal_outputs[c].x;
    normal_outputs[c].z=cap;
    if !finite(cap) { atomicOr(&status.fault,4u); return; }
    if m==0.0 || cap==0.0 { return; }
    let v=relative(c); let r=norm(v);
    if !finite(r) { atomicOr(&status.fault,4u); return; }
    var impulse=m*v;
    var d=vec3<f32>(m,0.0,m); var sliding=0.0;
    if r>cap/m {
        let unit=v/r; sliding=1.0;
        impulse=cap*unit;
        d=(cap/r)*vec3<f32>(unit.y*unit.y,-unit.x*unit.y,unit.x*unit.x);
    }
    let corrected=contacts[c].zw+impulse/m;
    if !all(abs(impulse)<=vec2<f32>(3.402823e38)) || !all(abs(corrected)<=vec2<f32>(3.402823e38))
        || !all(abs(d)<=vec3<f32>(3.402823e38)) { atomicOr(&status.fault,4u); return; }
    outputs[c]=vec4<f32>(-impulse,corrected);
    derivatives[c]=vec4<f32>(d,sliding);
}
fn assemble(i:u32, with_jacobian:bool) {
    let n=params.width;
    let count=contact_count();
    var value=vec2<f32>(0.0);
    for (var j=0u;j<n;j++) {
        if with_jacobian { jacobian[i*n+j]=mass[i*n+j]; }
        let delta=two_sum(rate[j],-free_rates[j]);
        let product=rounded(mass[i*n+j]*delta.x);
        let error=fma(mass[i*n+j],delta.x,-product)+mass[i*n+j]*(delta.y+rate_tail[j]);
        let sum=two_sum(value.x,product);
        value=two_sum(sum.x,value.y+sum.y+error);
    }
    for (var c=0u;c<count;c++) {
        if contacts[c].x==0.0 { continue; }
        let a=rows[c*n+i];
        let normal=normal_rows[c*n+i];
        let np=rounded(normal*normal_outputs[c].x);
        let ns=two_sum(value.x,np);
        value=two_sum(ns.x,value.y+ns.y+fma(normal,normal_outputs[c].x,-np));
        for (var axis=0u;axis<2u;axis++) {
            let p=rounded(-a[axis]*outputs[c][axis]);
            let sum=two_sum(value.x,p);
            value=two_sum(sum.x,value.y+sum.y+fma(-a[axis],outputs[c][axis],-p));
        }
        if with_jacobian {
            let d=derivatives[c].xyz;
            let weighted=vec2<f32>(dot(a,d.xy),dot(a,d.yz));
            var cross=0.0;
            if derivatives[c].w!=0.0 {
                let unit=-outputs[c].xy/normal_outputs[c].z;
                cross=contacts[c].y*normal_outputs[c].w*dot(a,unit);
            }
            for (var j=0u;j<n;j++) { jacobian[i*n+j]+=dot(weighted,rows[c*n+j])+(normal_outputs[c].w*normal+cross)*normal_rows[c*n+j]; }
        }
    }
    gradient[i]=rounded(value.x+value.y);
}
fn residual_scale(i:u32)->f32 {
    var scale=1.0;
    for (var j=0u;j<params.width;j++) { scale+=abs(mass[i*params.width+j]*free_rates[j]); }
    return scale;
}
fn newton_direction()->bool {
    let n=params.width;
    for (var i=0u;i<n;i++) {
        direction[i]=-gradient[i];
        for (var j=0u;j<n;j++) { lower[i*n+j]=jacobian[i*n+j]; }
    }
    for (var k=0u;k<n;k++) {
        var pivot=k;
        for (var i=k+1u;i<n;i++) { if abs(lower[i*n+k])>abs(lower[pivot*n+k]) { pivot=i; } }
        if !finite(lower[pivot*n+k]) || lower[pivot*n+k]==0.0 { return false; }
        if pivot!=k {
            for (var j=k;j<n;j++) { let temp=lower[k*n+j]; lower[k*n+j]=lower[pivot*n+j]; lower[pivot*n+j]=temp; }
            let rhs=direction[k]; direction[k]=direction[pivot]; direction[pivot]=rhs;
            status.pad+=1u;
        }
        for (var i=k+1u;i<n;i++) {
            let multiplier=lower[i*n+k]/lower[k*n+k];
            for (var j=k+1u;j<n;j++) { lower[i*n+j]-=multiplier*lower[k*n+j]; }
            direction[i]-=multiplier*direction[k];
        }
    }
    for (var k=0u;k<n;k++) {
        let i=n-1u-k; var value=direction[i];
        for (var j=i+1u;j<n;j++) { value-=lower[i*n+j]*direction[j]; }
        direction[i]=value/lower[i*n+i];
        if !finite(direction[i]) { return false; }
    }
    return true;
}
fn prepare(iteration:u32) {
    status.iterations=iteration+1u;
    let n=params.width;
    if !newton_direction() { atomicOr(&status.fault,32u); running=0u; return; }
    slope=0.0; merit=0.0;
    var converged=true;
    for (var i=0u;i<n;i++) {
        let scale=residual_scale(i);
        let residual=gradient[i]/scale;
        var predicted=0.0;
        for (var j=0u;j<n;j++) { predicted+=jacobian[i*n+j]*direction[j]; }
        merit+=residual*residual;
        slope+=2.0*residual*(predicted/scale);
        converged=converged && abs(gradient[i])<=1e-6*scale && abs(direction[i])<=1e-6*max(1.0,abs(rate[i]));
        base_rate[i]=rate[i]; base_tail[i]=rate_tail[i];
    }
    if !finite(merit) || !finite(slope) { atomicOr(&status.fault,4u); running=0u; return; }
    if converged {
        for (var i=0u;i<n;i++) { rates[i]=rounded(rate[i]+rate_tail[i]); }
        running=0u; return;
    }
    if slope>=0.0 { atomicOr(&status.fault,8u); running=0u; }
    step=1.0;
}
@compute @workgroup_size(64)
fn solve(@builtin(local_invocation_index) lane:u32) {
    rounding_lane=lane;
    if lane==0u {
        atomicStore(&active_end,0u);
        running=u32(atomicLoad(&status.fault)==0u); done=0u;
        status.iterations=0u; status.backtracks=0u; status.pad=0u;
        for (var i=0u;i<params.width;i++) {
            rate[i]=free_rates[i]; rate_tail[i]=0.0;
            if !finite(rate[i]) { atomicOr(&status.fault,1u); }
            for (var j=0u;j<params.width;j++) {
                let v=mass[i*params.width+j]; jacobian[i*params.width+j]=v;
                if !finite(v) || v!=mass[j*params.width+i] { atomicOr(&status.fault,2u); }
            }
        }
        if !factor() { atomicOr(&status.fault,2u); }
    }
    if workgroupUniformLoad(&running)==0u { return; }
    for (var c=lane;c<params.contacts;c+=64u) {
        let value=contacts[c];
        let normal=normal_data[c];
        outputs[c]=vec4<f32>(0.0,0.0,value.zw);
        derivatives[c]=vec4<f32>(0.0);
        normal_outputs[c]=vec4<f32>(0.0);
        if value.x==0.0 { contact(c); }
        if value.x>0.0 { atomicMax(&active_end,c+1u); }
        if !all(abs(normal)<=vec4<f32>(3.402823e38)) || normal.y<0.0 || any(normal.zw!=vec2<f32>(0.0)) { atomicOr(&status.fault,1u); }
        if !all(abs(value)<=vec4<f32>(3.402823e38)) || value.x<0.0 || value.y<0.0 { atomicOr(&status.fault,1u); }
        for (var i=0u;i<params.width;i++) {
            if !all(abs(rows[c*params.width+i])<=vec2<f32>(3.402823e38)) || !finite(normal_rows[c*params.width+i]) { atomicOr(&status.fault,1u); }
        }
    }
    storageBarrier();
    workgroupBarrier();
    let count=contact_count();
    for (var iteration=0u;iteration<params.iterations;iteration++) {
        for (var c=lane;c<count;c+=64u) { contact(c); }
        storageBarrier();
        if lane<params.width { assemble(lane,true); }
        workgroupBarrier();
        if lane==0u {
            if atomicLoad(&status.fault)==0u { prepare(iteration); } else { running=0u; }
        }
        if workgroupUniformLoad(&running)==0u { return; }
        for (var backtrack=0u;backtrack<32u;backtrack++) {
            if lane<params.width {
                let delta=two_sum(base_rate[lane],step*direction[lane]);
                let next=two_sum(delta.x,delta.y+base_tail[lane]);
                rate[lane]=next.x; rate_tail[lane]=next.y;
            }
            workgroupBarrier();
            for (var c=lane;c<count;c+=64u) { contact(c); }
            storageBarrier();
            if lane<params.width { assemble(lane,false); }
            workgroupBarrier();
            var value=0.0;
            if lane<params.width { let residual=gradient[lane]/residual_scale(lane); value=residual*residual; }
            reduction[lane]=value;
            workgroupBarrier();
            if lane==0u {
                var candidate=0.0;
                for (var i=0u;i<64u;i++) { candidate+=reduction[i]; }
                if atomicLoad(&status.fault)!=0u { running=0u; }
                else if finite(candidate) && candidate<=merit+1e-4*step*slope { done=1u; }
                else { step*=0.5; status.backtracks+=1u; }
            }
            if workgroupUniformLoad(&running)==0u { return; }
            if workgroupUniformLoad(&done)!=0u { break; }
        }
        if lane==0u {
            if done==0u { atomicOr(&status.fault,8u); running=0u; }
            done=0u;
        }
        if workgroupUniformLoad(&running)==0u { return; }
        storageBarrier();
    }
    if lane==0u { atomicOr(&status.fault,16u); }
}

