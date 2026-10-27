struct State { high:Six, low:Six }
struct NewtonStatus { fault:atomic<u32>, steps:u32, linear_iterations:u32, trials:u32, residual:f32, physical:f32, damping:f32, diagnostic_limit:u32 }
@group(0) @binding(13) var<storage,read> free_rates:array<vec2<f32>>;
@group(0) @binding(14) var<storage,read_write> states:array<State>;
@group(0) @binding(15) var<storage,read_write> saved:array<State>;
@group(0) @binding(16) var<storage,read_write> report:NewtonStatus;
struct Outer { stop:u32, accepted:u32, error:f32, physical:f32, releases:u32 }
var<workgroup> outer:Outer;
fn finite4(x:vec4<f32>)->bool {return all(abs(x)<=vec4<f32>(3.402823e38));}
fn six(k:vec3<f32>,w:vec3<f32>)->array<f32,6> {return array<f32,6>(k.x,k.y,k.z,w.x,w.y,w.z);}
fn physical_error(a:Local)->f32 {
    let machine=a.soil-a.relative;
    let separation=dot(a.normal,a.relative)+a.gap;
    let tangent=a.k-a.normal*a.lambda;
    let slip=a.relative-a.normal*dot(a.normal,a.relative);
    let ps=1.0+length(a.k)+length(a.w);
    let vs=1.0+length(a.soil)+length(machine);
    let ns=max(1.0,max(abs(dot(a.normal,a.soil)),max(abs(dot(a.normal,machine)),a.gap)));
    var error=max(0.0,max(-a.lambda/ps,max(-separation/ns,max(abs(a.lambda*separation)/(ps*vs),(length(tangent)-a.mu*a.lambda)/ps))));
    if length(slip)>1e-6*vs {error=max(error,length(tangent+a.mu*a.lambda*normalize(slip))/ps);}
    for (var i=0u;i<3u;i++) {
        if a.wall[i]==0.0 {error=max(error,abs(a.w[i])/ps);}
        else {error=max(error,max(-a.wall[i]*a.w[i]/ps,max(-a.wall[i]*a.soil[i]/vs,abs(a.w[i]*a.soil[i])/(ps*vs))));}
    }
    if !finite(error) || !finite(ps) || !finite(vs) {return 3.402823e38;}
    return error;
}
fn evaluate()->f32 {
    if lane<params.width {
        var sum=free_rates[lane];
        for (var slot=0u;slot<status.active_count;slot++) {
            let c=active_indices[slot];let x=states[c];
            for (var axis=0u;axis<3u;axis++) {sum=rate_add(sum,rate_mul(-response[c*params.width+lane][axis],vec2<f32>(x.high.k[axis],x.low.k[axis])));}
        }
        rates[lane]=sum;
        if !all(abs(sum)<=vec2<f32>(3.402823e38)) {atomicOr(&status.fault,2u);}
    }
    workgroupBarrier();
    var worst=0.0;
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];nonlinear_contact(c);worst=max(worst,vector[c].k.x);
        scratch[c]=packed(blocks[c].residual);
    }
    parts[lane]=vec2<f32>(worst,0.0);workgroupBarrier();
    if lane==0u {
        outer.physical=0.0;for (var i=0u;i<64u;i++) {outer.physical=max(outer.physical,parts[i].x);}
    }
    storageBarrier();let error=scratch_norm();
    if lane==0u {outer.error=error;report.residual=error;report.physical=outer.physical;}
    return workgroupUniformLoad(&outer.error);
}
fn prepare_linear() {
    if lane==0u {atomicStore(&status.fault,0u);atomicStore(&status.fallbacks,0u);status.iterations=0u;status.relative=0.0;status.beta=0.0;status.residual=0.0;status.pad=0.0;}
    storageBarrier();
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];direction[c]=Six();vector[c]=Six();scratch[c]=Six();inverses[c]=Inverse();
        if !build_inverse(c) {
            atomicAdd(&status.fallbacks,1u);
            for (var i=0u;i<6u;i++) {for (var j=0u;j<6u;j++) {inverses[c].a[6u*i+j]=select(0.0,1.0,i==j);}}
        }
    }
    storageBarrier();
}
fn trial(alpha:f32) {
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];var x=State();let base=saved[c];let d=direction[c];let wall=flags(c);
        for (var i=0u;i<3u;i++) {
            let k=rate_add(vec2<f32>(base.high.k[i],base.low.k[i]),rate_mul(alpha,vec2<f32>(d.k[i],0.0)));
            var w=vec2<f32>(0.0);if wall[i]!=0.0 {w=rate_add(vec2<f32>(base.high.w[i],base.low.w[i]),rate_mul(alpha,vec2<f32>(d.w[i],0.0)));}
            x.high.k[i]=k.x;x.low.k[i]=k.y;x.high.w[i]=w.x;x.low.w[i]=w.y;
        }
        states[c]=x;
    }
    storageBarrier();
}
fn restore() {
    for (var slot=lane;slot<status.active_count;slot+=64u) {let c=active_indices[slot];states[c]=saved[c];}
    storageBarrier();
}
fn regularized_direction_error(damping:f32,normal_scale:f32)->f32 {
    for (var slot=lane;slot<status.active_count;slot+=64u) {let c=active_indices[slot];vector[c]=direction[c];}
    storageBarrier();apply_operator();
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];var residual:array<f32,6>;
        for (var i=0u;i<6u;i++) {residual[i]=value(rate_add(vec2<f32>(component(scratch[c],i),0.0),vec2<f32>(blocks[c].residual[i],0.0)));}
        vector[c]=packed(residual);
    }
    storageBarrier();apply_transpose();
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];var gradient:array<f32,6>;
        for (var i=0u;i<6u;i++) {
            let penalty=rate_mul(damping,rate_mul(damping,vec2<f32>(component(direction[c],i),0.0)));
            gradient[i]=value(rate_add(vec2<f32>(component(scratch[c],i),0.0),penalty));
        }
        scratch[c]=packed(gradient);
    }
    storageBarrier();let error=scratch_norm();
    if lane==0u {
        status.residual=error;status.relative=error/normal_scale;
        if error<0.0 || normal_scale<=0.0 || !finite(normal_scale) || !finite(status.relative) {atomicOr(&status.fault,2u);status.relative=3.402823e38;}
        control.relative=status.relative;
    }
    return workgroupUniformLoad(&control.relative);
}
@compute @workgroup_size(64)
fn newton(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {atomicStore(&publication_status.publication_fault,1u);}
    for (var c=lane;c<max(params.contacts,1u);c+=64u) {publication_outputs[c]=PublicationOutput();}
    if lane<params.width {publication_rates[lane]=vec2<f32>(0.0);}
    storageBarrier();
    if lane==0u {outer=Outer();outer.stop=atomicLoad(&report.fault);}
    if workgroupUniformLoad(&outer.stop)!=0u {return;}
    if lane==0u {
        atomicStore(&status.fault,upstream[0]);report.steps=0u;report.linear_iterations=0u;report.trials=0u;report.damping=0.0;report.diagnostic_limit=0u;
        var count=0u;
        for (var c=0u;c<params.contacts;c++) {
            let mass=contacts[c].soil_mass.w;
            if !finite(mass) || mass<0.0 {atomicOr(&status.fault,1u);}
            if mass>0.0 {active_indices[count]=c;count++;}
        }
        status.active_count=count;
    }
    storageBarrier();
    for (var c=lane;c<max(params.contacts,1u);c+=64u) {
        direction[c]=Six();vector[c]=Six();scratch[c]=Six();
        if c>=params.contacts || contacts[c].soil_mass.w==0.0 {states[c]=State();continue;}
        let x=states[c];if !valid_six(x.high) || !valid_six(x.low) {atomicOr(&status.fault,1u);}
        for (var i=0u;i<params.width;i++) {
            let r=rows[c*params.width+i];let p=response[c*params.width+i];
            if !finite4(r) || !finite4(p) || r.w!=0.0 || p.w!=0.0 {atomicOr(&status.fault,1u);}
        }
    }
    if lane<params.width && !all(abs(free_rates[lane])<=vec2<f32>(3.402823e38)) {atomicOr(&status.fault,1u);}
    storageBarrier();
    if lane==0u {outer.stop=atomicLoad(&status.fault);if outer.stop!=0u {atomicOr(&report.fault,1u);}}
    if workgroupUniformLoad(&outer.stop)!=0u {return;}
    var merit=evaluate();
    for (var step=0u;step<=params.newton_iterations;step++) {
        if lane==0u {outer.stop=atomicLoad(&status.fault);if outer.stop!=0u || merit<0.0 || !finite(merit) {atomicOr(&report.fault,select(2u,outer.stop,outer.stop!=0u));outer.stop=1u;}}
        if workgroupUniformLoad(&outer.stop)!=0u {return;}
        if certify_candidate(lane) {return;}
        if lane==0u {outer.stop=atomicLoad(&publication_status.publication_fault)&3u;if outer.stop!=0u {atomicOr(&report.fault,outer.stop);}}
        if workgroupUniformLoad(&outer.stop)!=0u {return;}
        if step==params.newton_iterations {if lane==0u {atomicOr(&report.fault,8u);}return;}
        for (var slot=lane;slot<status.active_count;slot+=64u) {
            let c=active_indices[slot];saved[c]=states[c];scratch[c]=Six(states[c].high.k+states[c].low.k,states[c].high.w+states[c].low.w);
        }
        storageBarrier();let state_norm=scratch_norm();
        prepare_linear();solve(lane);
        if lane==0u {report.linear_iterations+=status.iterations;outer.stop=atomicLoad(&status.fault);outer.accepted=0u;}
        let invalid_direction=workgroupUniformLoad(&outer.stop)!=0u;
        if !invalid_direction {
            var alpha=1.0;
            for (var backtrack=0u;backtrack<16u;backtrack++) {
                if lane==0u {atomicStore(&status.fault,0u);report.trials++;}
                storageBarrier();trial(alpha);let candidate=evaluate();
                if lane==0u {outer.accepted=select(0u,1u,atomicLoad(&status.fault)==0u && candidate>=0.0 && candidate<merit*(1.0-1e-4*alpha));}
                if workgroupUniformLoad(&outer.accepted)!=0u {break;}
                alpha*=0.5;
            }
        }
        if workgroupUniformLoad(&outer.accepted)==0u {
            var damping=merit/(1.0+state_norm);
            for (var attempt=0u;attempt<8u;attempt++) {
                restore();if lane==0u {atomicStore(&status.fault,0u);}storageBarrier();evaluate();
                prepare_linear();least_squares(lane,sqrt(damping));
                if lane==0u {report.linear_iterations+=status.iterations;report.damping=damping;outer.stop=atomicLoad(&status.fault)&3u;}
                if workgroupUniformLoad(&outer.stop)==0u {
                    if lane==0u {atomicStore(&status.fault,0u);report.trials++;}storageBarrier();
                    trial(1.0);let candidate=evaluate();
                    if lane==0u {outer.accepted=select(0u,1u,atomicLoad(&status.fault)==0u && candidate>=0.0 && candidate<merit);}
                    if workgroupUniformLoad(&outer.accepted)!=0u {break;}
                }
                damping*=10.0;
            }
        }
        if workgroupUniformLoad(&outer.accepted)==0u {
            restore();if lane==0u {atomicStore(&status.fault,0u);}storageBarrier();evaluate();
            if lane==0u {atomicOr(&report.fault,16u);}return;
        }
        merit=workgroupUniformLoad(&outer.error);
        if lane==0u {report.steps=step+1u;}
    }
}
