struct Contact { soil_mass:vec4<f32>, normal_gap:vec4<f32>, friction:vec4<f32>, boundary:vec4<u32> }
struct Six { k:vec4<f32>, w:vec4<f32> }
struct Basis { v:Six, z:Six }
struct Block { residual:array<f32,6>, direct:array<f32,36>, machine:array<f32,18> }
struct Inverse { a:array<f32,36> }
struct Status { fault:atomic<u32>, iterations:u32, active_count:u32, fallbacks:atomic<u32>, relative:f32, beta:f32, residual:f32, pad:f32 }
struct Params { width:u32, contacts:u32, iterations:u32, tolerance:f32 }
@group(0) @binding(0) var<storage,read> blocks:array<Block>;
@group(0) @binding(1) var<storage,read> rows:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> response:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> contacts:array<Contact>;
@group(0) @binding(4) var<storage,read_write> inverses:array<Inverse>;
@group(0) @binding(5) var<storage,read_write> basis:array<Basis>;
@group(0) @binding(6) var<storage,read_write> vector:array<Six>;
@group(0) @binding(7) var<storage,read_write> scratch:array<Six>;
@group(0) @binding(8) var<storage,read_write> direction:array<Six>;
@group(0) @binding(9) var<storage,read_write> status:Status;
@group(0) @binding(10) var<uniform> params:Params;
@group(0) @binding(11) var<storage,read_write> active_indices:array<u32>;
@group(0) @binding(12) var<storage,read> upstream:array<u32>;
var<workgroup> h:array<f32,4160>;
var<workgroup> rotations:array<vec2<f32>,64>;
var<workgroup> g:array<f32,65>;
var<workgroup> coefficients:array<f32,64>;
var<workgroup> parts:array<vec2<f32>,64>;
var<workgroup> rates:array<vec2<f32>,32>;
struct Control { stop:u32, count:u32, scalar:f32, beta:f32, magnitude:f32, relative:f32 }
var<workgroup> control:Control;
var<private> lane:u32;
fn finite(x:f32)->bool { return abs(x)<=3.402823e38; }
fn value(x:vec2<f32>)->f32 { return rate_round(x.x+x.y); }
fn component(x:Six,i:u32)->f32 { if i<3u { return x.k[i]; } return x.w[i-3u]; }
fn packed(x:array<f32,6>)->Six { return Six(vec4<f32>(x[0],x[1],x[2],0.0),vec4<f32>(x[3],x[4],x[5],0.0)); }
fn finite_six(x:Six)->bool { return all(abs(x.k)<=vec4<f32>(3.402823e38)) && all(abs(x.w)<=vec4<f32>(3.402823e38)); }
fn scaled(x:Six,s:f32)->Six { return Six(vec4<f32>(x.k.xyz*s,0.0),vec4<f32>(x.w.xyz*s,0.0)); }
fn precondition(c:u32,x:Six)->Six {
    var result:array<f32,6>;
    for (var i=0u;i<6u;i++) {
        var sum=vec2<f32>(0.0);
        for (var j=0u;j<6u;j++) { sum=rate_add(sum,rate_mul(inverses[c].a[i*6u+j],vec2<f32>(component(x,j),0.0))); }
        result[i]=value(sum);
    }
    return packed(result);
}
fn build_inverse(c:u32)->bool {
    var mobility:array<vec3<f32>,3>;
    for (var axis=0u;axis<3u;axis++) { for (var j=0u;j<3u;j++) {
        var sum=vec2<f32>(0.0);
        for (var i=0u;i<params.width;i++) { sum=rate_add(sum,rate_mul(rows[c*params.width+i][axis],vec2<f32>(response[c*params.width+i][j],0.0))); }
        mobility[axis][j]=value(sum);
    } }
    var matrix:array<array<f32,12>,6>;
    for (var i=0u;i<6u;i++) { for (var j=0u;j<6u;j++) {
        var sum=vec2<f32>(blocks[c].direct[i*6u+j],0.0);
        if j<3u { for (var axis=0u;axis<3u;axis++) { sum=rate_add(sum,rate_mul(-blocks[c].machine[i*3u+axis],vec2<f32>(mobility[axis][j],0.0))); } }
        matrix[i][j]=value(sum); matrix[i][j+6u]=select(0.0,1.0,i==j);
        if !finite(matrix[i][j]) { atomicOr(&status.fault,2u); return false; }
    } }
    for (var k=0u;k<6u;k++) {
        var pivot=k; var largest=abs(matrix[k][k]);
        for (var i=k+1u;i<6u;i++) { if abs(matrix[i][k])>largest { largest=abs(matrix[i][k]); pivot=i; } }
        if !finite(largest) || largest<1e-20 { return false; }
        if pivot!=k { let row=matrix[k]; matrix[k]=matrix[pivot]; matrix[pivot]=row; }
        let diagonal=matrix[k][k];
        for (var j=0u;j<12u;j++) { matrix[k][j]/=diagonal; }
        for (var i=0u;i<6u;i++) { if i!=k {
            let ratio=matrix[i][k];
            for (var j=0u;j<12u;j++) { matrix[i][j]=value(rate_add(vec2<f32>(matrix[i][j],0.0),rate_mul(-ratio,vec2<f32>(matrix[k][j],0.0)))); }
        } }
    }
    for (var i=0u;i<6u;i++) { for (var j=0u;j<6u;j++) { if !finite(matrix[i][j+6u]) {return false;} } }
    for (var i=0u;i<6u;i++) { for (var j=0u;j<6u;j++) { inverses[c].a[6u*i+j]=matrix[i][j+6u]; } }
    return true;
}

@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x; rate_lane=c%64u;
    if c==0u {
        status.iterations=0u; status.active_count=0u; status.relative=0.0; status.beta=0.0; status.residual=0.0;
        if upstream[0]!=0u {atomicOr(&status.fault,1u);}
        var count=0u;
        for (var i=0u;i<params.contacts;i++) {if contacts[i].soil_mass.w>0.0 {active_indices[count]=i; count++;}}
        status.active_count=count;
    }
    if c>=max(params.contacts,1u) {return;}
    direction[c]=Six(); vector[c]=Six(); scratch[c]=Six(); inverses[c]=Inverse();
    if c>=params.contacts || atomicLoad(&status.fault)!=0u {return;}
    let mass=contacts[c].soil_mass.w;
    if !finite(mass) || mass<0.0 {atomicOr(&status.fault,1u);return;}
    if mass==0.0 {return;}
    for (var i=0u;i<6u;i++) {if !finite(blocks[c].residual[i]) {atomicOr(&status.fault,1u);return;}}
    for (var i=0u;i<36u;i++) {if !finite(blocks[c].direct[i]) {atomicOr(&status.fault,1u);return;}}
    for (var i=0u;i<18u;i++) {if !finite(blocks[c].machine[i]) {atomicOr(&status.fault,1u);return;}}
    for (var i=0u;i<params.width;i++) {
        let a=rows[c*params.width+i]; let b=response[c*params.width+i];
        if !all(abs(a)<=vec4<f32>(3.402823e38)) || !all(abs(b)<=vec4<f32>(3.402823e38)) || a.w!=0.0 || b.w!=0.0 {atomicOr(&status.fault,1u);return;}
    }
    if !build_inverse(c) {
        atomicAdd(&status.fallbacks,1u);
        for (var i=0u;i<6u;i++) {for (var j=0u;j<6u;j++) {inverses[c].a[6u*i+j]=select(0.0,1.0,i==j);}}
    }
}

fn load_vector(kind:u32,index:u32,c:u32)->Six {
    if kind==0u {return scratch[c];}
    return basis[index*max(params.contacts,1u)+c].v;
}
fn inner_product(a:u32,ai:u32,b:u32,bi:u32)->f32 {
    var sum=vec2<f32>(0.0);
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot]; let x=load_vector(a,ai,c); let y=load_vector(b,bi,c);
        for (var i=0u;i<6u;i++) {sum=rate_add(sum,rate_mul(component(x,i),vec2<f32>(component(y,i),0.0)));}
    }
    parts[lane]=sum; workgroupBarrier();
    if lane==0u {
        var total=vec2<f32>(0.0);
        for (var i=0u;i<64u;i++) {total=rate_add(total,parts[i]);}
        control.scalar=value(total);
    }
    return workgroupUniformLoad(&control.scalar);
}
fn apply_operator() {
    if lane<params.width {
        var sum=vec2<f32>(0.0);
        for (var slot=0u;slot<status.active_count;slot++) {
            let c=active_indices[slot];
            for (var axis=0u;axis<3u;axis++) {sum=rate_add(sum,rate_mul(-response[c*params.width+lane][axis],vec2<f32>(vector[c].k[axis],0.0)));}
        }
        rates[lane]=sum;
    }
    workgroupBarrier();
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot]; var point:array<vec2<f32>,3>;
        for (var axis=0u;axis<3u;axis++) {
            var sum=vec2<f32>(0.0);
            for (var i=0u;i<params.width;i++) {sum=rate_add(sum,rate_mul(rows[c*params.width+i][axis],rates[i]));}
            point[axis]=sum;
        }
        var result:array<f32,6>;
        for (var i=0u;i<6u;i++) {
            var sum=vec2<f32>(0.0);
            for (var j=0u;j<6u;j++) {sum=rate_add(sum,rate_mul(blocks[c].direct[6u*i+j],vec2<f32>(component(vector[c],j),0.0)));}
            for (var j=0u;j<3u;j++) {sum=rate_add(sum,rate_mul(blocks[c].machine[3u*i+j],point[j]));}
            result[i]=value(sum);
        }
        scratch[c]=packed(result);
    }
    storageBarrier();
}

fn apply_transpose() {
    if lane<params.width {
        var sum=vec2<f32>(0.0);
        for (var slot=0u;slot<status.active_count;slot++) {
            let c=active_indices[slot];
            for (var axis=0u;axis<3u;axis++) {
                var point=vec2<f32>(0.0);
                for (var i=0u;i<6u;i++) {point=rate_add(point,rate_mul(blocks[c].machine[i*3u+axis],vec2<f32>(component(vector[c],i),0.0)));}
                sum=rate_add(sum,rate_mul(rows[c*params.width+lane][axis],point));
            }
        }
        rates[lane]=sum;
    }
    workgroupBarrier();
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];var result:array<f32,6>;
        for (var i=0u;i<6u;i++) {
            var sum=vec2<f32>(0.0);
            for (var j=0u;j<6u;j++) {sum=rate_add(sum,rate_mul(blocks[c].direct[j*6u+i],vec2<f32>(component(vector[c],j),0.0)));}
            if i<3u {for (var j=0u;j<params.width;j++) {sum=rate_add(sum,rate_mul(-response[c*params.width+j][i],rates[j]));}}
            result[i]=value(sum);
        }
        scratch[c]=packed(result);
    }
    storageBarrier();
}

fn scratch_norm()->f32 {
    var largest=0.0; var invalid=false;
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let x=scratch[active_indices[slot]];
        for (var i=0u;i<6u;i++) {let v=component(x,i);if !finite(v) {invalid=true;}largest=max(largest,abs(v));}
    }
    parts[lane]=vec2<f32>(largest,select(0.0,1.0,invalid));workgroupBarrier();
    if lane==0u {
        var scale=0.0;var bad=false;
        for (var i=0u;i<64u;i++) {scale=max(scale,parts[i].x);bad=bad || parts[i].y!=0.0;}
        control.scalar=select(scale,-1.0,bad);
    }
    let scale=workgroupUniformLoad(&control.scalar);
    if scale<=0.0 {return scale;}
    var sum=vec2<f32>(0.0);
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let x=scratch[active_indices[slot]];
        for (var i=0u;i<6u;i++) {let v=component(x,i)/scale;sum=rate_add(sum,rate_mul(v,vec2<f32>(v,0.0)));}
    }
    parts[lane]=sum;workgroupBarrier();
    if lane==0u {
        var total=vec2<f32>(0.0);for (var i=0u;i<64u;i++) {total=rate_add(total,parts[i]);}
        control.scalar=scale*sqrt(max(value(total),0.0));
    }
    return workgroupUniformLoad(&control.scalar);
}
fn recover_direction(count:u32)->f32 {
    if lane==0u {
        status.iterations=count;
        for (var k=0u;k<count;k++) {
            let i=count-1u-k;var sum=vec2<f32>(g[i],0.0);
            for (var j=i+1u;j<count;j++) {sum=rate_add(sum,rate_mul(-h[i*64u+j],vec2<f32>(coefficients[j],0.0)));}
            coefficients[i]=value(sum)/h[i*64u+i];
            if !finite(coefficients[i]) {atomicOr(&status.fault,2u);}
        }
        control.magnitude=f32(atomicLoad(&status.fault));
    }
    if workgroupUniformLoad(&control.magnitude)!=0.0 {return 3.402823e38;}
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];var result:array<f32,6>;
        for (var i=0u;i<6u;i++) {
            var sum=vec2<f32>(0.0);
            for (var j=0u;j<count;j++) {sum=rate_add(sum,rate_mul(coefficients[j],vec2<f32>(component(basis[j*max(params.contacts,1u)+c].z,i),0.0)));}
            result[i]=value(sum);
        }
        let output=packed(result);
        if !finite_six(output) {atomicOr(&status.fault,2u);}
        direction[c]=output;vector[c]=output;
    }
    return true_direction_error();
}
fn true_direction_error()->f32 {
    for (var slot=lane;slot<status.active_count;slot+=64u) {let c=active_indices[slot];vector[c]=direction[c];}
    storageBarrier();apply_operator();
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];var result:array<f32,6>;
        for (var i=0u;i<6u;i++) {result[i]=value(rate_add(vec2<f32>(component(scratch[c],i),0.0),vec2<f32>(blocks[c].residual[i],0.0)));}
        scratch[c]=packed(result);
    }
    storageBarrier();let error=scratch_norm();
    if lane==0u {
        status.residual=error;status.relative=error/control.beta;
        if error<0.0 || !finite(status.relative) {atomicOr(&status.fault,2u);status.relative=3.402823e38;}
        control.relative=status.relative;
    }
    return workgroupUniformLoad(&control.relative);
}

@compute @workgroup_size(64)
fn solve(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {control=Control();control.stop=atomicLoad(&status.fault);}
    if workgroupUniformLoad(&control.stop)!=0u {return;}
    for (var i=lane;i<4160u;i+=64u) {h[i]=0.0;}
    for (var i=lane;i<65u;i+=64u) {g[i]=0.0;}
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];var rhs:array<f32,6>;
        for (var i=0u;i<6u;i++) {rhs[i]=-blocks[c].residual[i];}
        scratch[c]=packed(rhs);
    }
    storageBarrier();workgroupBarrier();
    let beta=scratch_norm();
    if lane==0u {control.beta=beta;status.beta=beta;g[0]=beta; if !finite(beta) || beta<0.0 {atomicOr(&status.fault,2u);control.stop=1u;}}
    if workgroupUniformLoad(&control.stop)!=0u || beta==0.0 {return;}
    for (var slot=lane;slot<status.active_count;slot+=64u) {let c=active_indices[slot];basis[c].v=scaled(scratch[c],1.0/beta);}
    storageBarrier();
    for (var j=0u;j<params.iterations;j++) {
        for (var slot=lane;slot<status.active_count;slot+=64u) {let c=active_indices[slot];let z=precondition(c,basis[j*max(params.contacts,1u)+c].v);basis[j*max(params.contacts,1u)+c].z=z;vector[c]=z;}
        storageBarrier();apply_operator();
        for (var orthogonal_pass=0u;orthogonal_pass<2u;orthogonal_pass++) {for (var i=0u;i<=j;i++) {
            let projection=inner_product(1u,i,0u,0u);
            if lane==0u {h[i*64u+j]+=projection;}
            for (var slot=lane;slot<status.active_count;slot+=64u) {
                let c=active_indices[slot];let base=basis[i*max(params.contacts,1u)+c].v;var result:array<f32,6>;
                for (var k=0u;k<6u;k++) {result[k]=value(rate_add(vec2<f32>(component(scratch[c],k),0.0),rate_mul(-projection,vec2<f32>(component(base,k),0.0))));}
                scratch[c]=packed(result);
            }
            storageBarrier();
        }}
        let magnitude=scratch_norm();
        if magnitude>0.0 && finite(magnitude) {for (var slot=lane;slot<status.active_count;slot+=64u) {let c=active_indices[slot];basis[(j+1u)*max(params.contacts,1u)+c].v=scaled(scratch[c],1.0/magnitude);}}
        if lane==0u {
            h[(j+1u)*64u+j]=magnitude;
            for (var i=0u;i<j;i++) {
                let a=h[i*64u+j];let b=h[(i+1u)*64u+j];let r=rotations[i];
                h[i*64u+j]=r.x*a+r.y*b;h[(i+1u)*64u+j]=-r.y*a+r.x*b;
            }
            let a=h[j*64u+j];let b=h[(j+1u)*64u+j];let scale=max(abs(a),abs(b));
            var length=0.0;if scale>0.0 {length=scale*sqrt((a/scale)*(a/scale)+(b/scale)*(b/scale));}
            if !finite(length) || !finite(magnitude) || magnitude<0.0 {atomicOr(&status.fault,2u);control.stop=1u;}
            else if length<1e-20 {control.stop=1u;}
            else {
                rotations[j]=vec2<f32>(a,b)/length;h[j*64u+j]=length;h[(j+1u)*64u+j]=0.0;
                g[j+1u]=-rotations[j].y*g[j];g[j]*=rotations[j].x;control.count=j+1u;
                status.pad=abs(g[j+1u])/beta;
                if magnitude==0.0 {control.stop=1u;}
            }
        }
        storageBarrier();
        let current_count=workgroupUniformLoad(&control.count);
        let true_error=recover_direction(current_count);
        if lane==0u && atomicLoad(&status.fault)!=0u {control.stop=1u;}
        if workgroupUniformLoad(&control.stop)!=0u || true_error<=params.tolerance {break;}
    }
    if lane==0u && status.relative>params.tolerance {atomicOr(&status.fault,8u);}
}

@compute @workgroup_size(64)
fn least_squares(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {control=Control();control.stop=atomicLoad(&status.fault);}
    if workgroupUniformLoad(&control.stop)!=0u {return;}
    let stride=max(params.contacts,1u);
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];var rhs:array<f32,6>;
        for (var i=0u;i<6u;i++) {rhs[i]=-blocks[c].residual[i];}
        scratch[c]=packed(rhs);
    }
    storageBarrier();let initial=scratch_norm();
    if lane==0u {control.beta=initial;status.beta=initial;status.residual=initial;status.relative=select(1.0,0.0,initial==0.0);}
    if initial==0.0 {return;}
    if initial<0.0 || !finite(initial) {if lane==0u {atomicOr(&status.fault,2u);}return;}
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];let u=scaled(scratch[c],1.0/initial);
        basis[c].v=u;vector[c]=u;
    }
    storageBarrier();apply_transpose();var alpha=scratch_norm();
    if alpha<=0.0 || !finite(alpha) {if lane==0u {atomicOr(&status.fault,select(8u,2u,alpha<0.0 || !finite(alpha)));}return;}
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];let v=scaled(scratch[c],1.0/alpha);
        basis[c].z=v;basis[stride+c].v=v;
    }
    var phibar=initial;var rhobar=alpha;
    storageBarrier();
    for (var iteration=0u;iteration<params.iterations;iteration++) {
        for (var slot=lane;slot<status.active_count;slot+=64u) {let c=active_indices[slot];vector[c]=basis[c].z;}
        storageBarrier();apply_operator();
        for (var slot=lane;slot<status.active_count;slot+=64u) {
            let c=active_indices[slot];var result:array<f32,6>;
            for (var i=0u;i<6u;i++) {result[i]=value(rate_add(vec2<f32>(component(scratch[c],i),0.0),rate_mul(-alpha,vec2<f32>(component(basis[c].v,i),0.0))));}
            scratch[c]=packed(result);
        }
        storageBarrier();let beta=scratch_norm();
        if beta<0.0 || !finite(beta) {if lane==0u {atomicOr(&status.fault,2u);}break;}
        for (var slot=lane;slot<status.active_count;slot+=64u) {
            let c=active_indices[slot];var u=Six();if beta>0.0 {u=scaled(scratch[c],1.0/beta);}
            basis[c].v=u;vector[c]=u;
        }
        storageBarrier();apply_transpose();
        for (var slot=lane;slot<status.active_count;slot+=64u) {
            let c=active_indices[slot];var result:array<f32,6>;
            for (var i=0u;i<6u;i++) {result[i]=value(rate_add(vec2<f32>(component(scratch[c],i),0.0),rate_mul(-beta,vec2<f32>(component(basis[c].z,i),0.0))));}
            scratch[c]=packed(result);
        }
        storageBarrier();alpha=scratch_norm();
        if alpha<0.0 || !finite(alpha) {if lane==0u {atomicOr(&status.fault,2u);}break;}
        let scale=max(abs(rhobar),beta);var rho=0.0;
        if scale>0.0 {rho=scale*sqrt((rhobar/scale)*(rhobar/scale)+(beta/scale)*(beta/scale));}
        if rho<=0.0 || !finite(rho) {if lane==0u {atomicOr(&status.fault,select(8u,2u,!finite(rho)));}break;}
        let cosine=rhobar/rho;let sine=beta/rho;let theta=sine*alpha;
        rhobar=-cosine*alpha;let phi=cosine*phibar;phibar*=sine;
        for (var slot=lane;slot<status.active_count;slot+=64u) {
            let c=active_indices[slot];var v=Six();if alpha>0.0 {v=scaled(scratch[c],1.0/alpha);}
            let w=basis[stride+c].v;var output:array<f32,6>;var next:array<f32,6>;
            for (var i=0u;i<6u;i++) {
                output[i]=value(rate_add(vec2<f32>(component(direction[c],i),0.0),rate_mul(phi/rho,vec2<f32>(component(w,i),0.0))));
                next[i]=value(rate_add(vec2<f32>(component(v,i),0.0),rate_mul(-theta/rho,vec2<f32>(component(w,i),0.0))));
            }
            direction[c]=packed(output);basis[stride+c].v=packed(next);basis[c].z=v;
        }
        storageBarrier();let error=true_direction_error();
        if lane==0u {status.iterations=iteration+1u;status.pad=abs(phibar)/initial;control.stop=atomicLoad(&status.fault);}
        if workgroupUniformLoad(&control.stop)!=0u || error<=params.tolerance || alpha==0.0 {break;}
    }
    if lane==0u && status.relative>params.tolerance {atomicOr(&status.fault,8u);}
}
