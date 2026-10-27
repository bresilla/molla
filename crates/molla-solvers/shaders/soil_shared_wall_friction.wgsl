struct SharedContact { soil_mass:vec4<f32>, normal_gap:vec4<f32>, friction:vec4<f32>, boundary:vec4<u32> }
struct SharedParams { width:u32, contacts:u32, sweeps:u32, iterations:u32 }
struct SharedStatus { fault:atomic<u32>, sweeps:u32, local_fault:atomic<u32>, contact:atomic<u32>, accelerated:u32, rejected:u32, attempted:u32, pad:u32, rates:array<vec2<f32>,32>, previous_f:array<f32,32>, control:vec4<u32> }
struct SharedHistory { previous_k:vec4<f32>, previous_w:vec4<f32>, baseline_k:vec4<f32>, baseline_w:vec4<f32> }
const SHARED_ACCELERATE:bool=true;
const SHARED_SEGMENTED:bool=true;
@group(0) @binding(0) var<storage,read> shared_mass:array<f32>;
@group(0) @binding(1) var<storage,read> shared_free:array<f32>;
@group(0) @binding(2) var<storage,read> shared_rows:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> shared_contacts:array<SharedContact>;
@group(0) @binding(4) var<storage,read_write> shared_lower:array<f32>;
@group(0) @binding(5) var<storage,read_write> shared_response:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> shared_inputs:array<Input>;
@group(0) @binding(7) var<storage,read_write> shared_outputs:array<Output>;
@group(0) @binding(8) var<storage,read_write> shared_rates:array<f32>;
@group(0) @binding(9) var<storage,read_write> shared_status:SharedStatus;
@group(0) @binding(10) var<uniform> shared_params:SharedParams;
@group(0) @binding(11) var<storage,read_write> shared_history:array<SharedHistory>;
var<private> shared_q:array<vec2<f32>,32>;
var<private> shared_previous_f:array<f32,32>;
fn shared_fault(flag:u32,c:u32,local:u32) {
    atomicOr(&shared_status.fault,flag);
    atomicOr(&shared_status.local_fault,local);
    atomicMax(&shared_status.contact,c);
}
@compute @workgroup_size(1)
fn shared_factor() {
    if atomicLoad(&shared_status.fault)!=0u { return; }
    let n=shared_params.width;
    shared_status.sweeps=0u;
    shared_status.accelerated=0u; shared_status.rejected=0u; shared_status.attempted=0u;
    shared_status.control=vec4<u32>(0u);
    for (var i=0u;i<n;i++) {
        if !finite(shared_free[i]) { shared_fault(1u,0u,0u); return; }
        shared_status.rates[i]=vec2<f32>(shared_free[i],0.0);
        shared_status.previous_f[i]=0.0;
        for (var j=0u;j<n;j++) {
            let value=shared_mass[i*n+j];
            if !finite(value) || value!=shared_mass[j*n+i] { shared_fault(2u,0u,0u); return; }
        }
    }
    for (var i=0u;i<n;i++) { for (var j=0u;j<=i;j++) {
        var value=shared_mass[i*n+j];
        for (var k=0u;k<j;k++) { value-=shared_lower[i*n+k]*shared_lower[j*n+k]; }
        if !finite(value) || (i==j && value<=0.0) { shared_fault(2u,0u,0u); return; }
        if i==j { shared_lower[i*n+j]=sqrt(value); }
        else { shared_lower[i*n+j]=value/shared_lower[j*n+j]; }
    } }
}
@compute @workgroup_size(64)
fn shared_prepare(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    rounding_slot=lane;
    let c=id.x; let n=shared_params.width;
    if c>=shared_params.contacts || atomicLoad(&shared_status.fault)!=0u { return; }
    shared_outputs[c]=Output();
    let node=shared_contacts[c];
    if !all(abs(node.soil_mass)<=vec4<f32>(3.402823e38)) || node.soil_mass.w<0.0
        || !all(abs(node.friction)<=vec4<f32>(3.402823e38)) || node.friction.x<0.0
        || any(node.friction.yzw!=vec3<f32>(0.0)) { shared_fault(1u,c+1u,0u); return; }
    for (var i=0u;i<n;i++) {
        let row=shared_rows[c*n+i];
        if !all(abs(row)<=vec4<f32>(3.402823e38)) || row.w!=0.0 { shared_fault(1u,c+1u,0u); return; }
    }
    if node.soil_mass.w==0.0 { return; }
    var response:array<vec3<f32>,32>;
    for (var i=0u;i<n;i++) {
        var value=shared_rows[c*n+i].xyz;
        for (var j=0u;j<i;j++) { value-=shared_lower[i*n+j]*response[j]; }
        response[i]=value/shared_lower[i*n+i];
    }
    for (var k=0u;k<n;k++) {
        let i=n-1u-k; var value=response[i];
        for (var j=i+1u;j<n;j++) { value-=shared_lower[j*n+i]*response[j]; }
        response[i]=value/shared_lower[i*n+i];
        if !all(abs(response[i])<=vec3<f32>(3.402823e38)) { shared_fault(4u,c+1u,0u); return; }
        shared_response[c*n+i]=vec4<f32>(response[i],0.0);
    }
    var a:mat3x3<f32>;
    for (var i=0u;i<3u;i++) { for (var j=0u;j<=i;j++) {
        var sum=vec2<f32>(0.0);
        for (var d=0u;d<n;d++) { sum=add_product(sum,shared_rows[c*n+d][i],response[d][j],0.0); }
        a[i][j]=rounded(sum.x+sum.y); a[j][i]=a[i][j];
    } }
    shared_inputs[c]=Input(vec4<f32>(a[0],0.0),vec4<f32>(a[1],0.0),vec4<f32>(a[2],0.0),
        vec4<f32>(0.0,0.0,0.0,node.friction.x),node.soil_mass,node.normal_gap,node.boundary);
}
fn shared_soil_velocity(c:u32,k:vec3<f32>,w:vec3<f32>)->vec3<f32> {
    let node=shared_contacts[c]; var soil=node.soil_mass.xyz+(k+w)/node.soil_mass.w;
    for (var axis=0u;axis<3u;axis++) {
        let flag=(node.boundary.x>>(2u*axis))&3u;
        if flag!=0u && select(1.0,-1.0,flag==2u)*w[axis]>0.0 { soil[axis]=0.0; }
    }
    return soil;
}
fn shared_merit_soil_velocity(c:u32,k:vec3<f32>,w:vec3<f32>)->vec3<f32> {
    let node=shared_contacts[c]; var soil=vec3<f32>(0.0);
    for (var axis=0u;axis<3u;axis++) {
        let impulse=two_sum(k[axis],w[axis]);
        let momentum=add_product(impulse,node.soil_mass.w,node.soil_mass[axis],0.0);
        soil[axis]=rounded(momentum.x/node.soil_mass.w+momentum.y/node.soil_mass.w);
    }
    return soil;
}
fn shared_velocity(c:u32)->vec3<f32> {
    var result=vec3<f32>(0.0);
    for (var axis=0u;axis<3u;axis++) {
        var sum=vec2<f32>(0.0);
        for (var i=0u;i<shared_params.width;i++) { sum=add_product(sum,shared_rows[c*shared_params.width+i][axis],shared_q[i].x,shared_q[i].y); }
        result[axis]=rounded(sum.x+sum.y);
    }
    return result;
}
fn shared_rebuild_rates() {
    let n=shared_params.width;
    for (var i=0u;i<n;i++) {
        var sum=vec2<f32>(shared_free[i],0.0);
        for (var c=0u;c<shared_params.contacts;c++) {
            if shared_contacts[c].soil_mass.w==0.0 { continue; }
            let p=shared_outputs[c].machine_impulse_normal.xyz;
            for (var axis=0u;axis<3u;axis++) { sum=add_product(sum,shared_response[c*n+i][axis],p[axis],0.0); }
        }
        shared_q[i]=sum;
    }
}
fn shared_natural_merit()->f32 {
    var value=vec2<f32>(0.0);
    for (var c=0u;c<shared_params.contacts;c++) {
        let node=shared_contacts[c]; let m=node.soil_mass.w; if m==0.0 { continue; }
        let k=-shared_outputs[c].machine_impulse_normal.xyz; let w=shared_outputs[c].wall_impulse.xyz;
        let soil=node.soil_mass.xyz+(k+w)/m; let normal=normalize(node.normal_gap.xyz);
        let relative=soil-shared_velocity(c); let lambda=dot(normal,k);
        let tangent=k-lambda*normal; let slip=relative-dot(normal,relative)*normal;
        let a=shared_inputs[c]; let rho=1.0/(1.0/m+a.a0.x+a.a1.y+a.a2.z);
        let scale=1.0+length(k)+length(w);
        let fnormal=(lambda-max(0.0,lambda-rho*(dot(normal,relative)+node.normal_gap.w)))/scale;
        let y=tangent-rho*slip; let radius=length(y); let cap=node.friction.x*max(lambda,0.0);
        var projected=y; if radius>cap { projected=(cap/radius)*y; }
        let ft=(tangent-projected)/scale;
        var sum=fnormal*fnormal+dot(ft,ft);
        for (var axis=0u;axis<3u;axis++) {
            let flag=(node.boundary.x>>(2u*axis))&3u;
            var fwall=w[axis];
            if flag!=0u {
                let sign=select(1.0,-1.0,flag==2u); let beta=sign*w[axis];
                fwall=beta-max(0.0,beta-m*sign*soil[axis]);
            }
            sum+=(fwall/scale)*(fwall/scale);
        }
        if !finite(sum) || !finite(scale) { return 3.402823e38; }
        value=two_sum(value.x,value.y+sum);
    }
    let result=value.x+value.y;
    if !finite(result) { return 3.402823e38; }
    return result;
}
fn shared_set_impulses(c:u32,p:vec4<f32>,w:vec4<f32>) {
    shared_outputs[c].machine_impulse_normal=p; shared_outputs[c].wall_impulse=w;
    let node=shared_contacts[c];
    let soil=shared_soil_velocity(c,-p.xyz,w.xyz);
    shared_outputs[c].soil_velocity_energy=vec4<f32>(soil,shared_outputs[c].soil_velocity_energy.w);
}
fn shared_accelerate(start:array<vec2<f32>,32>,has_previous:bool) {
    let n=shared_params.width; var current:array<f32,32>; var difference:array<f32,32>;
    for (var i=0u;i<n;i++) {
        let delta=two_sum(shared_q[i].x,-start[i].x);
        current[i]=rounded(delta.x+delta.y+shared_q[i].y-start[i].y);
        difference[i]=current[i]-shared_previous_f[i];
    }
    var numerator=vec2<f32>(0.0); var denominator=vec2<f32>(0.0);
    for (var i=0u;i<n;i++) { for (var j=0u;j<n;j++) {
        numerator=add_product(numerator,difference[i]*shared_mass[i*n+j],current[j],0.0);
        denominator=add_product(denominator,difference[i]*shared_mass[i*n+j],difference[j],0.0);
    } }
    let top=numerator.x+numerator.y; let bottom=denominator.x+denominator.y;
    var coefficient=0.0;
    if has_previous && finite(top) && finite(bottom) && bottom>1e-30 {
        coefficient=clamp(-top/bottom,-2.0,2.0);
        if !finite(coefficient) { coefficient=0.0; }
    }
    let baseline=shared_natural_merit();
    for (var c=0u;c<shared_params.contacts;c++) {
        if shared_contacts[c].soil_mass.w==0.0 { continue; }
        let p=shared_outputs[c].machine_impulse_normal; let w=shared_outputs[c].wall_impulse;
        shared_history[c].baseline_k=p; shared_history[c].baseline_w=w;
        if coefficient!=0.0 {
            let trial_p=p.xyz+coefficient*(p.xyz-shared_history[c].previous_k.xyz);
            let trial_w=w.xyz+coefficient*(w.xyz-shared_history[c].previous_w.xyz);
            let lambda=dot(-trial_p,normalize(shared_contacts[c].normal_gap.xyz));
            shared_set_impulses(c,vec4<f32>(trial_p,lambda),vec4<f32>(trial_w,0.0));
        }
    }
    if coefficient!=0.0 {
        shared_status.attempted+=1u;
        shared_rebuild_rates();
        if shared_natural_merit()>=baseline {
            shared_status.rejected+=1u;
            for (var c=0u;c<shared_params.contacts;c++) {
                if shared_contacts[c].soil_mass.w==0.0 { continue; }
                shared_set_impulses(c,shared_history[c].baseline_k,shared_history[c].baseline_w);
            }
            shared_rebuild_rates();
        } else { shared_status.accelerated+=1u; }
    }
    for (var c=0u;c<shared_params.contacts;c++) {
        shared_history[c].previous_k=shared_history[c].baseline_k;
        shared_history[c].previous_w=shared_history[c].baseline_w;
    }
    shared_previous_f=current;
}
fn shared_certify()->bool {
    let n=shared_params.width;
    var energy=vec2<f32>(0.0); var energy_scale=1.0;
    for (var i=0u;i<n;i++) {
        let qi=rounded(shared_q[i].x+shared_q[i].y);
        if !finite(qi) { shared_fault(4u,0u,0u); return false; }
        var momentum=vec2<f32>(0.0); var scale=1.0;
        for (var j=0u;j<n;j++) {
            let qj=rounded(shared_q[j].x+shared_q[j].y);
            let delta=two_sum(qj,-shared_free[j]);
            momentum=add_product(momentum,shared_mass[i*n+j],delta.x,delta.y);
            scale+=abs(shared_mass[i*n+j])*abs(qj-shared_free[j]);
            let delta_i=two_sum(qi,-shared_free[i]);
            let e=0.5*shared_mass[i*n+j]*((delta_i.x+delta_i.y)*qj+shared_free[i]*(delta.x+delta.y));
            energy=two_sum(energy.x,energy.y+e); energy_scale+=abs(e);
        }
        for (var c=0u;c<shared_params.contacts;c++) {
            if shared_contacts[c].soil_mass.w==0.0 { continue; }
            let k=-shared_outputs[c].machine_impulse_normal.xyz;
            for (var axis=0u;axis<3u;axis++) { momentum=add_product(momentum,shared_rows[c*n+i][axis],k[axis],0.0); }
            scale+=dot(abs(shared_rows[c*n+i].xyz),abs(k));
        }
        if !finite(scale) || !finite(momentum.x+momentum.y) { shared_fault(4u,0u,0u); return false; }
        if abs(momentum.x+momentum.y)>1e-6*scale { return false; }
    }
    for (var c=0u;c<shared_params.contacts;c++) {
        let node=shared_contacts[c]; if node.soil_mass.w==0.0 { continue; }
        let output=shared_outputs[c]; let k=-output.machine_impulse_normal.xyz; let w=output.wall_impulse.xyz;
        let soil=shared_soil_velocity(c,k,w);
        let machine=shared_velocity(c); let relative=soil-machine; let normal=normalize(node.normal_gap.xyz);
        let lambda=dot(k,normal); let tangent=k-lambda*normal;
        let separation=dot(normal,relative)+node.normal_gap.w;
        let slip=relative-dot(normal,relative)*normal; let cap=node.friction.x*lambda;
        let vs=1.0+length(soil)+length(machine)+node.normal_gap.w; let ps=1.0+length(k)+length(w);
        if !all(abs(soil)<=vec3<f32>(3.402823e38)) || !finite(vs) || !finite(ps) { shared_fault(4u,c+1u,0u); return false; }
        let momentum_error=node.soil_mass.w*(soil-node.soil_mass.xyz)-k-w;
        let momentum_scale=ps+length(node.soil_mass.w*node.soil_mass.xyz);
        if !finite(momentum_scale) || !all(abs(momentum_error)<=vec3<f32>(3.402823e38))
            || length(momentum_error)>1e-6*momentum_scale { return false; }
        if lambda< -1e-6*ps || separation< -1e-6*vs || abs(lambda*separation)>1e-6*ps*vs
            || length(tangent)>cap+1e-6*ps { return false; }
        if length(slip)>1e-6*vs && length(tangent+cap*normalize(slip))>1e-6*ps { return false; }
        for (var axis=0u;axis<3u;axis++) {
            let flag=(node.boundary.x>>(2u*axis))&3u;
            if flag==0u { if w[axis]!=0.0 { return false; } }
            else {
                let sign=select(1.0,-1.0,flag==2u);
                if sign*w[axis]< -1e-6*ps || sign*soil[axis]< -1e-6*vs || abs(w[axis]*soil[axis])>1e-6*ps*vs { return false; }
            }
        }
        let e=0.5*node.soil_mass.w*dot(soil-node.soil_mass.xyz,soil+node.soil_mass.xyz);
        energy=two_sum(energy.x,energy.y+e); energy_scale+=abs(e);
    }
    if !finite(energy.x+energy.y) || !finite(energy_scale) { shared_fault(4u,0u,0u); return false; }
    return energy.x+energy.y<=1e-6*energy_scale;
}
@compute @workgroup_size(64)
fn shared_solve(@builtin(local_invocation_index) lane:u32) {
    if lane!=0u || atomicLoad(&shared_status.fault)!=0u || shared_status.control.x!=0u { return; }
    rounding_slot=0u;
    let n=shared_params.width;
    for (var i=0u;i<n;i++) {
        shared_q[i]=shared_status.rates[i];
        shared_previous_f[i]=shared_status.previous_f[i];
    }
    let first=shared_status.sweeps;
    let last=select(shared_params.sweeps,first+1u,SHARED_SEGMENTED);
    for (var sweep=first;sweep<last;sweep++) {
        shared_status.sweeps=sweep+1u;
        let start=shared_q;
        for (var c=0u;c<shared_params.contacts;c++) {
            if shared_contacts[c].soil_mass.w==0.0 { continue; }
            let old_k=-shared_outputs[c].machine_impulse_normal.xyz;
            for (var i=0u;i<n;i++) {
                for (var axis=0u;axis<3u;axis++) { shared_q[i]=add_product(shared_q[i],shared_response[c*n+i][axis],old_k[axis],0.0); }
            }
            var input=shared_inputs[c]; input.machine_mu=vec4<f32>(shared_velocity(c),input.machine_mu.w);
            shared_inputs[c].machine_mu=input.machine_mu;
            solve_block(input,shared_params.iterations);
            if block_output.status.x!=0u { shared_fault(64u,c+1u,block_output.status.x); return; }
            shared_outputs[c]=block_output;
            let p=block_output.machine_impulse_normal.xyz;
            for (var i=0u;i<n;i++) {
                for (var axis=0u;axis<3u;axis++) { shared_q[i]=add_product(shared_q[i],shared_response[c*n+i][axis],p[axis],0.0); }
            }
        }
        shared_rebuild_rates();
        if SHARED_ACCELERATE { shared_accelerate(start,sweep>0u); }
        if shared_certify() {
            shared_status.control.x=1u;
            for (var i=0u;i<n;i++) { shared_rates[i]=rounded(shared_q[i].x+shared_q[i].y); }
            for (var c=0u;c<shared_params.contacts;c++) {
                if shared_contacts[c].soil_mass.w==0.0 { continue; }
                shared_outputs[c].machine_velocity_residual=vec4<f32>(shared_velocity(c),0.0);
                let node=shared_contacts[c]; let soil=shared_outputs[c].soil_velocity_energy.xyz;
                shared_outputs[c].soil_velocity_energy.w=0.5*node.soil_mass.w*dot(soil-node.soil_mass.xyz,soil+node.soil_mass.xyz);
                var bits=0u;
                for (var axis=0u;axis<3u;axis++) {
                    let flag=(node.boundary.x>>(2u*axis))&3u;
                    if flag!=0u && select(1.0,-1.0,flag==2u)*shared_outputs[c].wall_impulse[axis]>0.0 { bits|=1u<<axis; }
                }
                shared_outputs[c].status.w=bits;
            }
            return;
        }
        if atomicLoad(&shared_status.fault)!=0u { return; }
    }
    if shared_status.sweeps>=shared_params.sweeps {
        for (var i=0u;i<n;i++) { shared_status.rates[i]=shared_q[i]; }
        shared_fault(16u,0u,0u); return;
    }
    for (var i=0u;i<n;i++) {
        shared_status.rates[i]=shared_q[i];
        shared_status.previous_f[i]=shared_previous_f[i];
    }
}
