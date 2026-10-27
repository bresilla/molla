struct Contact { soil_mass:vec4<f32>, normal_gap:vec4<f32>, friction:vec4<f32>, boundary:vec4<u32> }
struct Params { width:u32, contacts:u32, pad0:u32, pad1:u32 }
struct Status { fault:atomic<u32>, contact:atomic<u32>, active_count:atomic<u32>, max_residual:atomic<u32>, pad:vec4<u32> }
@group(0) @binding(0) var<storage,read> mass:array<f32>;
@group(0) @binding(1) var<storage,read> rows:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> contacts:array<Contact>;
@group(0) @binding(3) var<storage,read> upstream:array<u32>;
@group(0) @binding(4) var<storage,read_write> lower:array<f32>;
@group(0) @binding(5) var<storage,read_write> response:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> correction:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> status:Status;
@group(0) @binding(8) var<uniform> params:Params;
@group(0) @binding(9) var<storage,read_write> failure:array<u32>;
fn finite(x:f32)->bool {return abs(x)<=3.402823e38;}
fn value(x:vec2<f32>)->f32 {return rate_round(x.x+x.y);}
fn fault(bits:u32,c:u32) {atomicOr(&status.fault,bits);if c>0u {atomicMax(&status.contact,c);}}
fn component_error(c:u32,i:u32,axis:u32)->f32 {
    let n=params.width;var residual=vec2<f32>(-rows[c*n+i][axis],0.0);var scale=abs(rows[c*n+i][axis]);
    for(var j=0u;j<n;j++) {
        let m=mass[i*n+j];let r=response[c*n+j][axis];
        residual=rate_add(residual,rate_mul(m,vec2<f32>(r,0.0)));scale+=abs(m)*abs(r);
    }
    let error=abs(value(residual))/select(scale,1.0,scale==0.0);
    return select(-1.0,error,finite(error)&&finite(scale));
}
fn certified(c:u32)->bool {
    for(var i=0u;i<params.width;i++) {for(var axis=0u;axis<3u;axis++) {
        let error=component_error(c,i,axis);if error<0.0 || error>1e-6 {return false;}
    }}
    return true;
}
@compute @workgroup_size(1)
fn factor() {
    rate_lane=0u;let n=params.width;
    for(var i=0u;i<n*n;i++) {lower[i]=0.0;}
    if atomicLoad(&status.fault)!=0u {return;}
    if upstream[0]==0u {failure[0]=0u;}
    atomicStore(&status.contact,0u);atomicStore(&status.active_count,0u);atomicStore(&status.max_residual,0u);
    if upstream[0]!=0u {fault(1u,0u);}
    if atomicLoad(&status.fault)!=0u {return;}
    for(var i=0u;i<n;i++) {for(var j=0u;j<n;j++) {
        let m=mass[i*n+j];if !finite(m) || m!=mass[j*n+i] {fault(2u,0u);return;}
    }}
    for(var i=0u;i<n;i++) {for(var j=0u;j<=i;j++) {
        var sum=vec2<f32>(mass[i*n+j],0.0);
        for(var k=0u;k<j;k++) {sum=rate_add(sum,rate_mul(-lower[i*n+k],vec2<f32>(lower[j*n+k],0.0)));}
        let d=value(sum);var l=0.0;
        if !finite(d) || (i==j && d<=0.0) {fault(2u,0u);return;}
        if i==j {l=sqrt(d);} else {l=d/lower[j*n+j];}
        if !finite(l) {fault(2u,0u);return;}
        lower[i*n+j]=l;
    }}
}
@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    rate_lane=lane;let c=id.x;let n=params.width;
    if c>=max(params.contacts,1u) {return;}
    for(var i=0u;i<n;i++) {response[c*n+i]=vec4<f32>(0.0);correction[c*n+i]=vec4<f32>(0.0);}
    if c>=params.contacts || atomicLoad(&status.fault)!=0u {return;}
    let weight=contacts[c].soil_mass.w;
    if !finite(weight) || weight<0.0 {fault(1u,c+1u);return;}
    if weight==0.0 {return;}
    atomicAdd(&status.active_count,1u);
    for(var i=0u;i<n;i++) {
        let row=rows[c*n+i];
        if !all(abs(row)<=vec4<f32>(3.402823e38)) || row.w!=0.0 {fault(1u,c+1u);return;}
        for(var axis=0u;axis<3u;axis++) {
            var sum=vec2<f32>(row[axis],0.0);
            for(var j=0u;j<i;j++) {sum=rate_add(sum,rate_mul(-lower[i*n+j],vec2<f32>(response[c*n+j][axis],0.0)));}
            response[c*n+i][axis]=value(sum)/lower[i*n+i];
        }
    }
    for(var k=0u;k<n;k++) {
        let i=n-1u-k;
        for(var axis=0u;axis<3u;axis++) {
            var sum=vec2<f32>(response[c*n+i][axis],0.0);
            for(var j=i+1u;j<n;j++) {sum=rate_add(sum,rate_mul(-lower[j*n+i],vec2<f32>(response[c*n+j][axis],0.0)));}
            let r=value(sum)/lower[i*n+i];response[c*n+i][axis]=r;
            if !finite(r) {fault(4u,c+1u);return;}
        }
    }
    for(var refinement=0u;refinement<8u;refinement++) {
        if refinement>=2u && certified(c) {break;}
        for(var i=0u;i<n;i++) {for(var axis=0u;axis<3u;axis++) {
            if refinement>=2u {
                let error=component_error(c,i,axis);
                if error>=0.0 && error<=1e-6 {correction[c*n+i][axis]=0.0;continue;}
            }
            var residual=vec2<f32>(rows[c*n+i][axis],0.0);
            for(var j=0u;j<n;j++) {residual=rate_add(residual,rate_mul(-mass[i*n+j],vec2<f32>(response[c*n+j][axis],0.0)));}
            correction[c*n+i][axis]=value(residual);
        }}
        for(var i=0u;i<n;i++) {for(var axis=0u;axis<3u;axis++) {
            var sum=vec2<f32>(correction[c*n+i][axis],0.0);
            for(var j=0u;j<i;j++) {sum=rate_add(sum,rate_mul(-lower[i*n+j],vec2<f32>(correction[c*n+j][axis],0.0)));}
            correction[c*n+i][axis]=value(sum)/lower[i*n+i];
        }}
        for(var k=0u;k<n;k++) {
            let i=n-1u-k;
            for(var axis=0u;axis<3u;axis++) {
                var sum=vec2<f32>(correction[c*n+i][axis],0.0);
                for(var j=i+1u;j<n;j++) {sum=rate_add(sum,rate_mul(-lower[j*n+i],vec2<f32>(correction[c*n+j][axis],0.0)));}
                correction[c*n+i][axis]=value(sum)/lower[i*n+i];
            }
        }
        for(var i=0u;i<n;i++) {for(var axis=0u;axis<3u;axis++) {
            response[c*n+i][axis]=value(rate_sum(response[c*n+i][axis],correction[c*n+i][axis]));
            if !finite(response[c*n+i][axis]) {fault(4u,c+1u);return;}
        }}
    }
    for(var i=0u;i<n;i++) {for(var axis=0u;axis<3u;axis++) {
        let error=component_error(c,i,axis);
        if error<0.0 {fault(4u,c+1u);return;}
        atomicMax(&status.max_residual,bitcast<u32>(error));
        if error>1e-6 {fault(4u,c+1u);return;}
    }}
}
@compute @workgroup_size(64)
fn finish(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x;if c>=max(params.contacts,1u) || atomicLoad(&status.fault)==0u {return;}
    if c==0u && failure[0]==0u {
        let n=params.width;let selected=atomicLoad(&status.contact);
        failure[0]=1u;failure[1]=n;failure[2]=params.contacts;failure[3]=selected;
        failure[4]=atomicLoad(&status.fault);failure[5]=selected;
        failure[6]=atomicLoad(&status.active_count);failure[7]=atomicLoad(&status.max_residual);
        for(var i=8u;i<16u;i++) {failure[i]=0u;}
        failure[12]=upstream[0];
        for(var i=0u;i<n*n;i++) {failure[16u+i]=bitcast<u32>(mass[i]);}
        let base=16u+n*n;
        for(var i=0u;i<4u*n+16u;i++) {failure[base+i]=0u;}
        if selected>0u && selected<=params.contacts {
            let slot=selected-1u;
            for(var i=0u;i<n;i++) {for(var a=0u;a<4u;a++) {failure[base+4u*i+a]=bitcast<u32>(rows[slot*n+i][a]);}}
            let contact=contacts[slot];let offset=base+4u*n;
            for(var a=0u;a<4u;a++) {
                failure[offset+a]=bitcast<u32>(contact.soil_mass[a]);
                failure[offset+4u+a]=bitcast<u32>(contact.normal_gap[a]);
                failure[offset+8u+a]=bitcast<u32>(contact.friction[a]);
                failure[offset+12u+a]=contact.boundary[a];
            }
        }
    }
    for(var i=0u;i<params.width;i++) {response[c*params.width+i]=vec4<f32>(0.0);correction[c*params.width+i]=vec4<f32>(0.0);}
}
