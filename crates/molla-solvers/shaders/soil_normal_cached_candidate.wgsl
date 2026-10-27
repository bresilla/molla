const WIDTH: u32 = 32u;
const HARD_WIDTH: u32 = 1u;
const HARD_ROWS: bool = false;
struct Params { width:u32, contacts:u32, iterations:u32, pad:u32 }
struct Status { fault:atomic<u32>, iterations:u32, backtracks:u32, pad:u32 }
@group(0) @binding(0) var<storage,read> mass:array<f32>;
@group(0) @binding(1) var<storage,read> free_rates:array<f32>;
@group(0) @binding(2) var<storage,read> rows:array<f32>;
@group(0) @binding(3) var<storage,read> contacts:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> rates:array<f32>;
@group(0) @binding(5) var<storage,read_write> impulses:array<f32>;
@group(0) @binding(6) var<storage,read_write> normal_velocities:array<f32>;
@group(0) @binding(7) var<storage,read_write> status:Status;
@group(0) @binding(8) var<uniform> params:Params;
var<workgroup> hessian:array<f32,WIDTH*WIDTH>;
var<workgroup> lower:array<f32,WIDTH*WIDTH>;
var<workgroup> rate_value:array<f32,WIDTH>;
var<workgroup> rate_tail:array<f32,WIDTH>;
var<workgroup> arithmetic_round:array<atomic<u32>,64>;
var<private> rounding_lane:u32;
var<workgroup> has_hard:atomic<u32>;
var<workgroup> contact_count:atomic<u32>;
var<workgroup> cached_contacts:array<u32,256>;
var<workgroup> cached_count:u32;
var<workgroup> cache_offsets:array<u32,64>;
var<workgroup> active_count:u32;
var<workgroup> solver_running:u32;
var<workgroup> gradient:array<f32,WIDTH>;
var<workgroup> direction:array<f32,WIDTH>;
var<workgroup> publication:array<vec3<f32>,WIDTH>;
var<workgroup> iteration_gate:vec2<f32>;
var<workgroup> responses:array<f32,HARD_WIDTH*HARD_WIDTH>;
var<workgroup> schur:array<f32,HARD_WIDTH*HARD_WIDTH>;
var<workgroup> working_rows:array<u32,HARD_WIDTH>;
var<workgroup> row_norm:array<f32,HARD_WIDTH>;
var<workgroup> multipliers:array<f32,HARD_WIDTH>;

fn finite(x:f32)->bool { return abs(x)<=3.402823e38; }
fn cache_contacts(lane:u32) {
    let end=atomicLoad(&contact_count);
    let span=(end+63u)/64u;
    let begin=min(lane*span,end);
    let stop=min(begin+span,end);
    var count=0u;
    for (var c=begin;c<stop;c++) {
        if contacts[c].x>0.0 || contacts[c].w==1.0 { count++; }
    }
    cache_offsets[lane]=count;
    workgroupBarrier();
    if lane==0u {
        var offset=0u;
        for (var i=0u;i<64u;i++) {
            let next=cache_offsets[i];
            cache_offsets[i]=offset;
            offset+=next;
        }
        cached_count=offset;
    }
    workgroupBarrier();
    var offset=cache_offsets[lane];
    for (var c=begin;c<stop;c++) {
        if contacts[c].x>0.0 || contacts[c].w==1.0 {
            if cached_count<=256u { cached_contacts[offset]=c; }
            offset++;
        }
    }
    workgroupBarrier();
}
fn contact_limit()->u32 {
    if cached_count<=256u { return cached_count; }
    return atomicLoad(&contact_count);
}
fn contact_index(index:u32)->u32 {
    if cached_count<=256u { return cached_contacts[index]; }
    return index;
}
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
fn machine_rate(i:u32)->f32 {
    if rate_tail[i]==0.0 { return rate_value[i]; }
    return rounded(rate_value[i]+rate_tail[i]);
}
fn contact_pair(c:u32)->vec2<f32> {
    var value=vec2<f32>(0.0);
    for (var i=0u;i<params.width;i++) {
        let a=rows[c*params.width+i];
        let q=rate_value[i]; let tail=rate_tail[i];
        let product=rounded(a*q);
        let error=fma(a,q,-product)+a*tail;
        let sum=two_sum(value.x,product);
        value=two_sum(sum.x,value.y+sum.y+error);
    }
    let offset=two_sum(value.x,-contacts[c].y);
    value=two_sum(offset.x,value.y+offset.y);
    let gap=two_sum(value.x,-contacts[c].z);
    return two_sum(gap.x,value.y+gap.y);
}
fn violation(c:u32)->f32 { return normal_velocities[c]; }
fn finite_impulse(c:u32)->f32 { return impulses[c]; }
fn impulse_from_pair(c:u32,value:vec2<f32>)->f32 {
    if value.x+value.y<=0.0 { return 0.0; }
    let product=rounded(contacts[c].x*value.x);
    return rounded(product+(fma(contacts[c].x,value.x,-product)+contacts[c].x*value.y));
}
fn assemble_publication(i:u32) {
    let n=params.width;
    let contact_end=contact_limit();
    var residual=vec2<f32>(0.0); var scale=0.0;
    for (var j=0u;j<n;j++) {
        let difference=two_sum(machine_rate(j),-free_rates[j]);
        let product=rounded(mass[i*n+j]*difference.x);
        let error=fma(mass[i*n+j],difference.x,-product)+mass[i*n+j]*difference.y;
        let sum=two_sum(residual.x,product);
        residual=two_sum(sum.x,residual.y+sum.y+error);
        scale+=abs(mass[i*n+j]*free_rates[j]);
    }
    for (var index=0u;index<contact_end;index++) {
        let c=contact_index(index);
        if contacts[c].x==0.0 { continue; }
        let impulse=finite_impulse(c);
        let product=rounded(rows[c*n+i]*impulse);
        let sum=two_sum(residual.x,product);
        residual=two_sum(sum.x,residual.y+sum.y+fma(rows[c*n+i],impulse,-product));
    }
    publication[i]=vec3<f32>(residual,scale);
}
fn published_momentum_converged(count:u32)->bool {
    let n=params.width;
    for (var i=0u;i<n;i++) {
        var residual=publication[i].xy; let scale=publication[i].z;
        for (var c=0u;c<count;c++) {
            let impulse=multipliers[c]/row_norm[c];let a=rows[working_rows[c]*n+i];
            let product=rounded(a*impulse);let sum=two_sum(residual.x,product);
            residual=two_sum(sum.x,residual.y+sum.y+fma(a,impulse,-product));
        }
        if !finite(residual.x) || !finite(residual.y) || !finite(scale) || abs(residual.x+residual.y)>1e-6*max(1.0,scale) { return false; }
    }
    return true;
}
fn factor()->bool {
    let n=params.width;
    for (var i=0u;i<n;i++) {
        for (var j=0u;j<=i;j++) {
            var value=hessian[i*n+j];
            for (var k=0u;k<j;k++) { value-=lower[i*n+k]*lower[j*n+k]; }
            if !finite(value) { return false; }
            if i==j {
                if value<=0.0 { return false; }
                lower[i*n+j]=sqrt(value);
            } else { lower[i*n+j]=value/lower[j*n+j]; }
        }
    }
    return true;
}
fn constrained_direction(count:u32)->bool {
    let n=params.width;
    for (var c=0u;c<count;c++) {
        var rhs=0.0;
        for (var i=0u;i<n;i++) {
            let a=rows[working_rows[c]*n+i]/row_norm[c];
            rhs+=a*direction[i];
            var value=a;
            for (var j=0u;j<i;j++) { value-=lower[i*n+j]*responses[c*n+j]; }
            responses[c*n+i]=value/lower[i*n+i];
        }
        multipliers[c]=rhs;
        for (var k=0u;k<n;k++) {
            let i=n-1u-k;
            var value=responses[c*n+i];
            for (var j=i+1u;j<n;j++) { value-=lower[j*n+i]*responses[c*n+j]; }
            responses[c*n+i]=value/lower[i*n+i];
        }
    }
    for (var c=0u;c<count;c++) {
        for (var d=0u;d<=c;d++) {
            var value=0.0;
            for (var i=0u;i<n;i++) { value+=(rows[working_rows[c]*n+i]/row_norm[c])*responses[d*n+i]; }
            for (var k=0u;k<d;k++) { value-=schur[c*n+k]*schur[d*n+k]; }
            if !finite(value) { return false; }
            if c==d {
                if value<=0.0 { return false; }
                schur[c*n+d]=sqrt(value);
            } else { schur[c*n+d]=value/schur[d*n+d]; }
        }
        var value=multipliers[c];
        for (var d=0u;d<c;d++) { value-=schur[c*n+d]*multipliers[d]; }
        multipliers[c]=value/schur[c*n+c];
    }
    for (var k=0u;k<count;k++) {
        let c=count-1u-k;
        var value=multipliers[c];
        for (var d=c+1u;d<count;d++) { value-=schur[d*n+c]*multipliers[d]; }
        multipliers[c]=value/schur[c*n+c];
        if !finite(multipliers[c]) { return false; }
    }
    for (var i=0u;i<n;i++) {
        for (var c=0u;c<count;c++) { direction[i]-=responses[c*n+i]*multipliers[c]; }
    }
    return true;
}
fn energy_difference(step:f32,slope:f32)->f32 {
    let n=params.width;
    let contact_end=contact_limit();
    var difference=step*slope;
    for (var i=0u;i<n;i++) {
        var product=0.0;
        for (var j=0u;j<n;j++) { product+=mass[i*n+j]*direction[j]; }
        difference+=0.5*step*step*direction[i]*product;
    }
    for (var index=0u;index<contact_end;index++) {
        let c=contact_index(index);
        if contacts[c].x==0.0 { continue; }
        var change=0.0;
        for (var i=0u;i<n;i++) { change+=rows[c*n+i]*direction[i]; }
        let z=violation(c);
        let next=z+step*change;
        var curvature=0.0;
        if z>0.0 {
            if next>0.0 { curvature=step*step*change*change; }
            else { curvature=z*(-z-2.0*step*change); }
        } else if next>0.0 { curvature=next*next; }
        difference+=0.5*contacts[c].x*curvature;
    }
    return difference;
}

fn initialize()->bool {
    if atomicLoad(&status.fault)!=0u { return false; }
    let n=params.width;
    status.iterations=0u; status.backtracks=0u; status.pad=0u;
    for (var i=0u;i<n;i++) {
        if !finite(free_rates[i]) { atomicOr(&status.fault,1u); return false; }
        rate_value[i]=free_rates[i];
        rate_tail[i]=0.0;
        for (var j=0u;j<n;j++) {
            let value=mass[i*n+j];
            if !finite(value) || value!=mass[j*n+i] { atomicOr(&status.fault,2u); return false; }
            hessian[i*n+j]=value;
        }
    }
    if !factor() { atomicOr(&status.fault,2u); return false; }
    atomicStore(&has_hard,0u);
    atomicStore(&contact_count,0u);
    active_count=0u;
    return true;
}
fn validate_contacts(lane:u32) {
    let n=params.width;
    for (var c=lane;c<params.contacts;c+=64u) {
        impulses[c]=0.0; normal_velocities[c]=0.0;
        let input=contacts[c];
        let hard=HARD_ROWS && input.w==1.0 && input.x==0.0 && input.y==0.0;
        if !all(abs(input)<=vec4<f32>(3.402823e38)) || input.x<0.0 || input.z<0.0 || (input.w!=0.0 && !hard) {
            atomicOr(&status.fault,1u);
        }
        if hard { atomicOr(&has_hard,1u); }
        if input.x>0.0 || hard { atomicMax(&contact_count,c+1u); }
        for (var i=0u;i<n;i++) {
            if !finite(rows[c*n+i]) { atomicOr(&status.fault,1u); }
        }
    }
}
fn assemble(i:u32) {
    let n=params.width;
    let contact_end=contact_limit();
    var value=vec2<f32>(0.0);
    for (var j=0u;j<n;j++) {
        hessian[i*n+j]=mass[i*n+j];
        let difference=two_sum(rate_value[j],-free_rates[j]);
        let product=rounded(mass[i*n+j]*difference.x);
        let error=fma(mass[i*n+j],difference.x,-product)+mass[i*n+j]*(difference.y+rate_tail[j]);
        let sum=two_sum(value.x,product);
        value=two_sum(sum.x,value.y+sum.y+error);
    }
    for (var index=0u;index<contact_end;index++) {
        let c=contact_index(index);
        if contacts[c].x==0.0 { continue; }
        let z=violation(c);
        if z<=0.0 { continue; }
        let impulse=finite_impulse(c);
        let weighted=contacts[c].x*rows[c*n+i];
        let product=rounded(rows[c*n+i]*impulse);
        let sum=two_sum(value.x,product);
        value=two_sum(sum.x,value.y+sum.y+fma(rows[c*n+i],impulse,-product));
        for (var j=0u;j<n;j++) { hessian[i*n+j]+=weighted*rows[c*n+j]; }
    }
    gradient[i]=rounded(value.x+value.y);
}
fn prepare_iteration(iteration:u32)->bool {
    let n=params.width;
    let contact_end=contact_limit();
    status.iterations=iteration+1u;
    if !factor() { atomicOr(&status.fault,2u); return false; }
    for (var i=0u;i<n;i++) {
        var value=-gradient[i];
        for (var j=0u;j<i;j++) { value-=lower[i*n+j]*direction[j]; }
        direction[i]=value/lower[i*n+i];
    }
    var change=0.0; var scale=1.0; var slope=0.0;
    for (var k=0u;k<n;k++) {
        let i=n-1u-k;
        var value=direction[i];
        for (var j=i+1u;j<n;j++) { value-=lower[j*n+i]*direction[j]; }
        direction[i]=value/lower[i*n+i];
        if !finite(direction[i]) || !finite(gradient[i]) { atomicOr(&status.fault,4u); return false; }
    }
    if atomicLoad(&has_hard)!=0u && !constrained_direction(active_count) { atomicOr(&status.fault,64u); return false; }
    for (var i=0u;i<n;i++) {
        if !finite(direction[i]) { atomicOr(&status.fault,4u); return false; }
        change=max(change,abs(direction[i])); scale=max(scale,abs(machine_rate(i)));
        slope+=gradient[i]*direction[i];
    }
    var residual_converged=true;
    for (var i=0u;i<n;i++) {
        var momentum_scale=0.0;
        for (var j=0u;j<n;j++) { momentum_scale+=abs(mass[i*n+j]*free_rates[j]); }
        var residual=gradient[i];
        for (var c=0u;c<active_count;c++) { residual+=(rows[working_rows[c]*n+i]/row_norm[c])*multipliers[c]; }
        residual_converged=residual_converged && abs(residual)<=1e-6*max(1.0,momentum_scale);
    }
    var impulse_converged=true;
    for (var index=0u;index<contact_end;index++) {
        let c=contact_index(index);
        if contacts[c].x==0.0 { continue; }
        var delta=0.0;
        for (var i=0u;i<n;i++) { delta+=rows[c*n+i]*direction[i]; }
        let z=violation(c);
        let current=max(0.0,z); let next=max(0.0,z+delta);
        let difference=select(next-current,delta,z>0.0 && z+delta>0.0);
        let uncertainty=contacts[c].x*abs(difference);
        if !finite(uncertainty) { atomicOr(&status.fault,4u); return false; }
        impulse_converged=impulse_converged && uncertainty<=1e-6*max(1.0,contacts[c].x*current);
    }
    iteration_gate=vec2<f32>(slope,select(0.0,1.0,change<=1e-6*scale && impulse_converged && residual_converged));
    return true;
}
fn iterate()->bool {
    let n=params.width;
    let contact_end=contact_limit();
    let slope=iteration_gate.x;
    var needs_removal=false;
    for (var c=0u;c<active_count;c++) { needs_removal=needs_removal || multipliers[c]<0.0; }
    if iteration_gate.y!=0.0 && (needs_removal || published_momentum_converged(active_count)) {
        var remove=active_count; var worst=0.0;
        for (var c=0u;c<active_count;c++) {
            if multipliers[c]<worst { worst=multipliers[c]; remove=c; }
        }
        if remove<active_count {
            for (var c=remove;c+1u<active_count;c++) { working_rows[c]=working_rows[c+1u]; row_norm[c]=row_norm[c+1u]; }
            status.pad+=65536u;
            active_count-=1u; return true;
        }
        for (var index=0u;index<contact_end;index++) {
            let c=contact_index(index);
            if contacts[c].w!=1.0 { continue; }
            var bound_scale=max(1.0,contacts[c].z);
            for (var i=0u;i<n;i++) { bound_scale+=abs(rows[c*n+i]*machine_rate(i)); }
            let residual=violation(c);
            if !finite(residual) || !finite(bound_scale) { atomicOr(&status.fault,4u); return false; }
            if residual>1e-6*bound_scale { atomicOr(&status.fault,64u); return false; }
        }
        for (var c=0u;c<active_count;c++) {
            if !finite(multipliers[c]/row_norm[c]) { atomicOr(&status.fault,4u); return false; }
        }
        for (var c=0u;c<active_count;c++) { impulses[working_rows[c]]=multipliers[c]/row_norm[c]; }
        for (var i=0u;i<n;i++) { rates[i]=machine_rate(i); }
        return false;
    }
    var step=1.0; var blocker=params.contacts;
    if atomicLoad(&has_hard)!=0u {
        for (var index=0u;index<contact_end;index++) {
            let c=contact_index(index);
            if contacts[c].w!=1.0 { continue; }
            var included=false;
            for (var j=0u;j<active_count;j++) { included=included || working_rows[j]==c; }
            if included { continue; }
            var delta=0.0; var magnitude=0.0;
            for (var i=0u;i<n;i++) {
                let a=rows[c*n+i]; let value=a*direction[i];
                delta+=value; magnitude+=abs(value);
                for (var j=0u;j<active_count;j++) { magnitude+=abs(a*responses[j*n+i]*multipliers[j]); }
            }
            if !finite(delta) || !finite(magnitude) { atomicOr(&status.fault,4u); return false; }
            if delta<=1e-6*magnitude { continue; }
            let residual=violation(c);
            if !finite(residual) { atomicOr(&status.fault,4u); return false; }
            let candidate=max(0.0,-residual)/delta;
            if candidate<step { step=candidate; blocker=c; }
        }
    }
    if step==0.0 && blocker<params.contacts {
        if active_count>=n { atomicOr(&status.fault,64u); return false; }
        var norm=0.0;
        for (var i=0u;i<n;i++) { let a=rows[blocker*n+i]; norm+=a*a; }
        if !finite(norm) || norm<=0.0 { atomicOr(&status.fault,64u); return false; }
        working_rows[active_count]=blocker; row_norm[active_count]=sqrt(norm); active_count+=1u; status.pad+=1u;
        return true;
    }
    if !finite(slope) || slope>=0.0 { atomicOr(&status.fault,8u); return false; }
    let blocking_step=step;
    var accepted=false;
    for (var trial=0u;trial<24u;trial++) {
        let difference=energy_difference(step,slope);
        if finite(difference) && difference<=0.0001*step*slope {
            accepted=true; break;
        }
        status.backtracks+=1u;
        step*=0.5;
    }
    if !accepted { atomicOr(&status.fault,8u); return false; }
    for (var i=0u;i<n;i++) {
        let sum=two_sum(rate_value[i],step*direction[i]);
        let corrected=two_sum(sum.x,rate_tail[i]+sum.y);
        rate_value[i]=corrected.x; rate_tail[i]=corrected.y;
        if !finite(rate_value[i]) || !finite(rate_tail[i]) { atomicOr(&status.fault,4u); return false; }
    }
    if step==blocking_step && blocker<params.contacts {
        if active_count>=n { atomicOr(&status.fault,64u); return false; }
        var norm=0.0;
        for (var i=0u;i<n;i++) { let a=rows[blocker*n+i]; norm+=a*a; }
        if !finite(norm) || norm<=0.0 { atomicOr(&status.fault,64u); return false; }
        working_rows[active_count]=blocker; row_norm[active_count]=sqrt(norm); active_count+=1u; status.pad+=1u;
    }
    return true;
}

@compute @workgroup_size(64)
fn solve(@builtin(local_invocation_index) lane:u32) {
    rounding_lane=lane;
    if lane==0u { solver_running=u32(initialize()); }
    if workgroupUniformLoad(&solver_running)==0u { return; }
    validate_contacts(lane);
    storageBarrier();
    workgroupBarrier();
    if lane==0u {
        solver_running=u32(atomicLoad(&status.fault)==0u);
        if atomicLoad(&has_hard)!=0u {
            for (var i=0u;i<params.width;i++) { rate_value[i]=0.0; }
        }
    }
    if workgroupUniformLoad(&solver_running)==0u { return; }
    cache_contacts(lane);
    let contact_end=contact_limit();
    for (var iteration=0u;iteration<params.iterations;iteration++) {
        for (var index=lane;index<contact_end;index+=64u) {
            let c=contact_index(index);
            if contacts[c].x>0.0 || contacts[c].w==1.0 {
                let value=contact_pair(c);
                let z=rounded(value.x+value.y);
                let impulse=impulse_from_pair(c,value);
                if !finite(z) || !finite(impulse) { atomicOr(&status.fault,4u); }
                normal_velocities[c]=z; impulses[c]=impulse;
            }
        }
        storageBarrier();
        if lane<params.width { assemble(lane); }
        workgroupBarrier();
        if lane==0u {
            if atomicLoad(&status.fault)==0u { solver_running=u32(prepare_iteration(iteration)); }
            else { solver_running=0u; }
        }
        if workgroupUniformLoad(&solver_running)==0u { return; }
        if iteration_gate.y!=0.0 && lane<params.width { assemble_publication(lane); }
        workgroupBarrier();
        if lane==0u { solver_running=u32(iterate()); }
        let running=workgroupUniformLoad(&solver_running);
        storageBarrier();
        if running==0u { return; }
    }
    if lane==0u { atomicOr(&status.fault,16u); }
}

@compute @workgroup_size(64)
fn recover(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x;
    if c>=params.contacts || atomicLoad(&status.fault)!=0u { return; }
    if contacts[c].w==1.0 { normal_velocities[c]=0.0; return; }
    let impulse=impulses[c];
    let velocity=contacts[c].y+max(0.0,normal_velocities[c]);
    if !finite(impulse) || !finite(velocity) { atomicOr(&status.fault,4u); return; }
    normal_velocities[c]=velocity;
}
