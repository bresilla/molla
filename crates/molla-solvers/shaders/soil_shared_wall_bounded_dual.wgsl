var<workgroup> dual_base:array<vec2<f32>,32>;
var<workgroup> dual_current:array<vec2<f32>,32>;
var<workgroup> dual_trial:array<f32,32>;

struct DualProjection { value:vec3<f32>, derivative:mat3x3<f32> }
fn dual_outer(v:vec3<f32>)->mat3x3<f32> {
    return mat3x3<f32>(v*v.x,v*v.y,v*v.z);
}
fn dual_momentum(i:u32)->vec2<f32> {
    let n=shared_params.width; var value=represented_momentum(i); var scale=1.0;
    for (var j=0u;j<n;j++) {
        let q=rounded(shared_q[j].x+shared_q[j].y);
        scale+=abs(shared_mass[i*n+j])*abs(q-shared_free[j]);
    }
    for (var slot=0u;slot<shared_status.control.y;slot++) {
        let c=shared_history[slot].active_index;
        scale+=abs(dot(shared_rows[c*n+i].xyz,shared_outputs[c].machine_impulse_normal.xyz));
    }
    value.y=scale; return value;
}
fn dual_ball_plane(v:vec3<f32>,cap:f32,normal:vec3<f32>,lambda:f32)->DualProjection {
    let identity=mat3x3<f32>(vec3<f32>(1.0,0.0,0.0),vec3<f32>(0.0,1.0,0.0),vec3<f32>(0.0,0.0,1.0));
    let size=length(v); var scale=1.0;
    if size>cap { scale=cap/size; }
    var p=scale*v; var derivative=scale*identity;
    if dot(p,normal)< -lambda {
        let tangent=v-dot(v,normal)*normal; let size=length(tangent);
        let radius=sqrt(max(0.0,cap*cap-lambda*lambda));
        scale=1.0; if size>radius { scale=radius/size; }
        p=-lambda*normal+scale*tangent;
        derivative=scale*(identity-dual_outer(normal));
        if size>radius { derivative-=scale*dual_outer(tangent/size); }
    } else if size>cap { derivative-=scale*dual_outer(v/size); }
    return DualProjection(p,derivative);
}

fn dual_evaluate(hessian:bool)->vec2<f32> {
    let n=shared_params.width; var objective=0.0;
    for (var i=0u;i<n;i++) {
        represented_rhs[i]=dual_base[i].x;
        objective+=0.98e-6*dual_base[i].y*abs(dual_trial[i])-dual_base[i].x*dual_trial[i];
        if hessian { for (var j=0u;j<n;j++) { represented_gram[i*n+j]=0.0; } }
    }
    for (var slot=0u;slot<shared_status.control.y;slot++) {
        let c=shared_history[slot].active_index;
        shared_history[c].correction=vec4<f32>(0.0);
        if !represented_active(c) { continue; }
        let mass=shared_contacts[c].soil_mass.w;
        var v=vec3<f32>(0.0);
        for (var i=0u;i<n;i++) { v-=mass*dual_trial[i]*shared_rows[c*n+i].xyz; }
        let old=-shared_outputs[c].machine_impulse_normal.xyz;
        let wall=shared_outputs[c].wall_impulse.xyz;
        let soil=shared_outputs[c].soil_velocity_energy.xyz;
        let machine=shared_velocity(c);
        let cap=0.9999990463256836*min(0.25e-6*(1.0+length(old)+length(wall)),0.25e-6*mass*(1.0+length(soil)+length(machine)));
        let normal=normalize(shared_contacts[c].normal_gap.xyz);
        let projection=dual_ball_plane(v,cap,normal,dot(old,normal));
        let p=projection.value;
        if !all(abs(p)<=vec3<f32>(3.402823e38)) { return vec2<f32>(3.402823e38); }
        shared_history[c].correction=vec4<f32>(p,0.0);
        objective+=(dot(v,p)-0.5*dot(p,p))/mass;
        for (var i=0u;i<n;i++) {
            let row=shared_rows[c*n+i].xyz;
            represented_rhs[i]+=dot(row,p);
            if hessian {
                for (var j=0u;j<=i;j++) {
                    let value=mass*dot(row,projection.derivative*shared_rows[c*n+j].xyz);
                    represented_gram[i*n+j]+=value;
                    if i!=j { represented_gram[j*n+i]+=value; }
                }
            }
        }
    }
    var worst=0.0;
    for (var i=0u;i<n;i++) { worst=max(worst,abs(represented_rhs[i])/(1e-6*dual_base[i].y)); }
    return vec2<f32>(objective,worst);
}

fn represented_bounded_dual()->bool {
    let n=shared_params.width;
    for (var i=0u;i<n;i++) {
        dual_base[i]=dual_momentum(i);
        if !all(abs(dual_base[i])<=vec2<f32>(3.402823e38)) { return false; }
        dual_current[i]=vec2<f32>(0.0); dual_trial[i]=0.0;
    }
    for (var iteration=0u;iteration<8u;iteration++) {
        let value=dual_evaluate(true);
        if !all(abs(value)<=vec2<f32>(3.402823e38)) { return false; }
        if value.y<=1.0 { return true; }
        for (var i=0u;i<n;i++) {
            var direction=0.0;
            if dual_current[i].x!=0.0 { direction=sign(dual_current[i].x); }
            else if abs(represented_rhs[i])>0.98e-6*dual_base[i].y { direction=sign(represented_rhs[i]); }
            dual_current[i].y=direction;
            represented_rhs[i]-=0.98e-6*dual_base[i].y*direction;
            represented_order[i]=i; represented_weights[i]=0.0;
        }
        for (var i=0u;i<n;i++) {
            if dual_current[i].y==0.0 {
                represented_rhs[i]=0.0;
                for (var j=0u;j<n;j++) { represented_gram[i*n+j]=0.0; represented_gram[j*n+i]=0.0; }
            }
        }
        for (var i=0u;i<n;i++) {
            let diagonal=represented_gram[i*n+i];
            if !finite(diagonal) || diagonal<0.0 { return false; }
            represented_scale[i]=1.0;
            if diagonal>0.0 { represented_scale[i]=sqrt(diagonal); }
        }
        if !represented_factor_solve() { return false; }
        var fraction=1.0; var accepted=false;
        for (var attempt=0u;attempt<8u;attempt++) {
            for (var i=0u;i<n;i++) {
                let q=dual_current[i].x+fraction*represented_weights[i];
                dual_trial[i]=select(0.0,q,dual_current[i].y*q>0.0);
            }
            let candidate=dual_evaluate(false);
            if all(abs(candidate)<=vec2<f32>(3.402823e38)) && candidate.x<value.x {
                for (var i=0u;i<n;i++) { dual_current[i].x=dual_trial[i]; }
                accepted=true; break;
            }
            fraction*=0.5;
        }
        if !accepted { return false; }
    }
    let value=dual_evaluate(false);
    return all(abs(value)<=vec2<f32>(3.402823e38)) && value.y<=1.0;
}
