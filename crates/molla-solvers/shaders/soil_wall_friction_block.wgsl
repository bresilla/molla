struct Input { a0:vec4<f32>, a1:vec4<f32>, a2:vec4<f32>, machine_mu:vec4<f32>, soil_mass:vec4<f32>, normal_gap:vec4<f32>, boundary:vec4<u32> }
const LOCAL_FRICTION_TOLERANCE:f32=1e-5;
struct Output { machine_impulse_normal:vec4<f32>, soil_velocity_energy:vec4<f32>, machine_velocity_residual:vec4<f32>, wall_impulse:vec4<f32>, status:vec4<u32> }
var<private> block_output:Output;
var<private> h:array<array<f32,6>,6>;
var<private> jac:array<array<f32,6>,6>;
var<private> b:array<f32,6>;
var<private> rho:array<f32,6>;
var<private> axes:array<vec3<f32>,6>;
var<private> mu:f32;
var<private> linear_ok:bool;
var<private> point_inverse:mat3x3<f32>;
var<private> free_soil:vec3<f32>;
var<private> free_machine:vec3<f32>;
var<private> soil_mass:f32;
var<private> contact_gap:f32;
var<workgroup> arithmetic_round:array<atomic<u32>,64>;
var<private> rounding_slot:u32;
fn rounded(value:f32)->f32 {
    atomicStore(&arithmetic_round[rounding_slot],bitcast<u32>(value));
    return bitcast<f32>(atomicAdd(&arithmetic_round[rounding_slot],0u));
}
fn two_sum(a:f32,b:f32)->vec2<f32> {
    let sum=rounded(a+b); let vb=rounded(sum-a);
    let error_a=rounded(a-rounded(sum-vb)); let error_b=rounded(b-vb);
    return vec2<f32>(sum,rounded(error_a+error_b));
}
fn add_product(value:vec2<f32>,a:f32,hi:f32,lo:f32)->vec2<f32> {
    let product=rounded(a*hi); let error=fma(a,hi,-product)+a*lo;
    let sum=two_sum(value.x,product);
    return two_sum(sum.x,value.y+sum.y+error);
}
fn impulse_pair(axis:u32,first:u32,last:u32,x:array<f32,6>,tail:array<f32,6>)->vec2<f32> {
    var value=vec2<f32>(0.0);
    for (var j=first;j<last;j++) { value=add_product(value,axes[j][axis],x[j],tail[j]); }
    return value;
}
fn finite(x:f32)->bool { return abs(x)<=3.402823e38; }
fn norm(v:vec2<f32>)->f32 {
    let s=max(abs(v.x),abs(v.y));
    if s==0.0 { return 0.0; }
    return s*length(v/s);
}
fn residual(x:array<f32,6>, tail:array<f32,6>, build:bool)->array<f32,6> {
    var velocity:array<f32,6>; var f:array<f32,6>;
    var k:array<vec2<f32>,3>; var soil=vec3<f32>(0.0); var machine=vec3<f32>(0.0);
    for (var axis=0u;axis<3u;axis++) {
        k[axis]=impulse_pair(axis,0u,3u,x,tail);
        let w=impulse_pair(axis,3u,6u,x,tail);
        let sum=two_sum(k[axis].x,w.x);
        let force=two_sum(sum.x,sum.y+k[axis].y+w.y);
        let momentum=add_product(force,soil_mass,free_soil[axis],0.0);
        soil[axis]=rounded(momentum.x/soil_mass+momentum.y/soil_mass);
    }
    for (var axis=0u;axis<3u;axis++) {
        var value=vec2<f32>(free_machine[axis],0.0);
        for (var j=0u;j<3u;j++) { value=add_product(value,-point_inverse[j][axis],k[j].x,k[j].y); }
        machine[axis]=rounded(value.x+value.y);
    }
    for (var i=0u;i<6u;i++) {
        velocity[i]=dot(axes[i],soil);
        if i<3u { velocity[i]=dot(axes[i],soil-machine); }
        if i==0u { velocity[i]+=contact_gap; }
    }
    for (var i=0u;i<6u;i++) {
        if i==1u || i==2u { continue; }
        let enabled=i==0u || dot(axes[i],axes[i])>0.0;
        let is_active=enabled && x[i]-rho[i]*velocity[i]>0.0;
        f[i]=rounded(x[i]+tail[i]);
        if is_active { f[i]=rho[i]*velocity[i]; }
        if build {
            for (var j=0u;j<6u;j++) {
                jac[i][j]=select(select(0.0,1.0,i==j),rho[i]*h[i][j],is_active);
            }
        }
    }
    let y=vec2<f32>(x[1]-rho[1]*velocity[1],x[2]-rho[1]*velocity[2]);
    let radius=norm(y); let cap=mu*max(x[0],0.0);
    if radius<=cap {
        f[1]=rho[1]*velocity[1]; f[2]=rho[1]*velocity[2];
        if build { for (var i=1u;i<3u;i++) { for (var j=0u;j<6u;j++) { jac[i][j]=rho[1]*h[i][j]; } } }
    } else {
        let unit=y/radius;
        let projection=(cap/radius)*vec3<f32>(unit.y*unit.y,-unit.x*unit.y,unit.x*unit.x);
        let fx=two_sum(x[1],-cap*unit.x); let fy=two_sum(x[2],-cap*unit.y);
        f[1]=rounded(fx.x+fx.y+tail[1]); f[2]=rounded(fy.x+fy.y+tail[2]);
        if build { for (var j=0u;j<6u;j++) {
            let dy=vec2<f32>(select(0.0,1.0,j==1u)-rho[1]*h[1][j],select(0.0,1.0,j==2u)-rho[1]*h[2][j]);
            let dc=select(0.0,mu,j==0u && x[0]>=0.0);
            jac[1][j]=select(0.0,1.0,j==1u)-dot(projection.xy,dy)-dc*unit.x;
            jac[2][j]=select(0.0,1.0,j==2u)-dot(projection.yz,dy)-dc*unit.y;
        } }
    }
    return f;
}
fn merit(f:array<f32,6>)->f32 {
    var result=0.0;
    for (var i=0u;i<6u;i++) { let value=f[i]/(1.0+abs(rho[i]*b[i])); result+=value*value; }
    return result;
}
fn direction(f:array<f32,6>)->array<f32,6> {
    var rhs:array<f32,6>;
    for (var i=0u;i<6u;i++) { rhs[i]=-f[i]; }
    linear_ok=true;
    for (var k=0u;k<6u;k++) {
        var pivot=k;
        for (var i=k+1u;i<6u;i++) { if abs(jac[i][k])>abs(jac[pivot][k]) { pivot=i; } }
        if !finite(jac[pivot][k]) || jac[pivot][k]==0.0 { linear_ok=false; return rhs; }
        if pivot!=k {
            for (var j=k;j<6u;j++) { let t=jac[k][j]; jac[k][j]=jac[pivot][j]; jac[pivot][j]=t; }
            let t=rhs[k]; rhs[k]=rhs[pivot]; rhs[pivot]=t;
        }
        for (var i=k+1u;i<6u;i++) {
            let factor=jac[i][k]/jac[k][k];
            for (var j=k+1u;j<6u;j++) { jac[i][j]-=factor*jac[k][j]; }
            rhs[i]-=factor*rhs[k];
        }
    }
    for (var k=0u;k<6u;k++) {
        let i=5u-k;
        for (var j=i+1u;j<6u;j++) { rhs[i]-=jac[i][j]*rhs[j]; }
        rhs[i]/=jac[i][i];
        if !finite(rhs[i]) { linear_ok=false; }
    }
    return rhs;
}
fn small_physical_correction(x:array<f32,6>,delta:array<f32,6>)->bool {
    let k=axes[0]*x[0]+axes[1]*x[1]+axes[2]*x[2];
    let w=axes[3]*x[3]+axes[4]*x[4]+axes[5]*x[5];
    let dk=axes[0]*delta[0]+axes[1]*delta[1]+axes[2]*delta[2];
    let dw=axes[3]*delta[3]+axes[4]*delta[4]+axes[5]*delta[5];
    let soil=free_soil+(k+w)/soil_mass; let machine=free_machine-point_inverse*k;
    return length(dk)+length(dw)<=1e-6*(1.0+length(k)+length(w))
        && length((dk+dw)/soil_mass)+length(point_inverse*dk)<=1e-6*(1.0+length(soil)+length(machine));
}
fn publish(input:Input,x:array<f32,6>,tail:array<f32,6>,a:mat3x3<f32>,error:f32) {
    let mass=input.soil_mass.w;
    var k=vec3<f32>(0.0); var w=vec3<f32>(0.0);
    for (var axis=0u;axis<3u;axis++) {
        let kp=impulse_pair(axis,0u,3u,x,tail); let wp=impulse_pair(axis,3u,6u,x,tail);
        k[axis]=rounded(kp.x+kp.y); w[axis]=rounded(wp.x+wp.y);
    }
    let raw_soil=input.soil_mass.xyz+(k+w)/mass;
    var soil=raw_soil;
    for (var axis=0u;axis<3u;axis++) {
        if ((input.boundary.x>>(2u*axis))&3u)!=0u && x[3u+axis]>0.0 { soil[axis]=0.0; }
    }
    let machine=input.machine_mu.xyz-a*k;
    let relative=soil-machine;
    let separation=dot(axes[0],relative)+input.normal_gap.w;
    let slip=vec2<f32>(dot(axes[1],relative),dot(axes[2],relative));
    let tangent=vec2<f32>(x[1],x[2]); let cap=mu*x[0];
    let velocity_scale=1.0+length(soil)+length(machine)+input.normal_gap.w;
    let impulse_scale=1.0+length(k)+length(w);
    let momentum_error=mass*(soil-input.soil_mass.xyz)-k-w;
    let momentum_scale=impulse_scale+length(mass*input.soil_mass.xyz);
    let energy=dot(k,input.soil_mass.xyz-input.machine_mu.xyz)+dot(w,input.soil_mass.xyz)
        +0.5*dot(k,a*k)+0.5*dot(k+w,k+w)/mass+0.5*mass*dot(soil-raw_soil,soil+raw_soil);
    if !all(abs(soil)<=vec3<f32>(3.402823e38)) || !all(abs(machine)<=vec3<f32>(3.402823e38))
        || !all(abs(k)<=vec3<f32>(3.402823e38)) || !all(abs(w)<=vec3<f32>(3.402823e38)) || !finite(energy) {
        block_output.status.x=4u; return;
    }
    var valid=finite(momentum_scale) && all(abs(momentum_error)<=vec3<f32>(3.402823e38))
        && length(momentum_error)<=1e-6*momentum_scale
        && x[0]>=0.0 && separation>=-1e-5*velocity_scale
        && abs(x[0]*separation)<=1e-5*impulse_scale*velocity_scale
        && norm(tangent)<=cap+1e-5*impulse_scale
        && energy<=1e-5*impulse_scale*velocity_scale;
    if norm(slip)>LOCAL_FRICTION_TOLERANCE*velocity_scale { valid=valid && norm(tangent+cap*slip/norm(slip))<=LOCAL_FRICTION_TOLERANCE*impulse_scale; }
    var wall_bits=0u;
    for (var i=3u;i<6u;i++) {
        let speed=dot(axes[i],soil);
        valid=valid && x[i]>=0.0 && speed>=-1e-5*velocity_scale
            && abs(x[i]*speed)<=1e-5*impulse_scale*velocity_scale;
        if x[i]>0.0 { wall_bits|=1u<<(i-3u); }
    }
    if !valid { block_output.status.x=32u; return; }
    block_output.machine_impulse_normal=vec4<f32>(-k,x[0]);
    block_output.soil_velocity_energy=vec4<f32>(soil,energy);
    block_output.machine_velocity_residual=vec4<f32>(machine,error);
    block_output.wall_impulse=vec4<f32>(w,0.0);
    block_output.status.x=0u; block_output.status.w=wall_bits;
}
fn solve_block(input:Input,iterations:u32) {
    block_output=Output(vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<u32>(1u,0u,0u,0u));
    let mask=input.boundary.x;
    if !all(abs(input.a0)<=vec4<f32>(3.402823e38)) || !all(abs(input.a1)<=vec4<f32>(3.402823e38))
        || !all(abs(input.a2)<=vec4<f32>(3.402823e38)) || !all(abs(input.machine_mu)<=vec4<f32>(3.402823e38))
        || !all(abs(input.soil_mass)<=vec4<f32>(3.402823e38)) || !all(abs(input.normal_gap)<=vec4<f32>(3.402823e38))
        || input.a0.w!=0.0 || input.a1.w!=0.0 || input.a2.w!=0.0 || input.machine_mu.w<0.0
        || input.soil_mass.w<=0.0 || input.normal_gap.w<0.0 || mask>=64u || any(input.boundary.yzw!=vec3<u32>(0u))
        || abs(dot(input.normal_gap.xyz,input.normal_gap.xyz)-1.0)>1e-5 { return; }
    let a=mat3x3<f32>(input.a0.xyz,input.a1.xyz,input.a2.xyz);
    if a[0][1]!=a[1][0] || a[0][2]!=a[2][0] || a[1][2]!=a[2][1] { block_output.status.x=2u; return; }
    let scale=max(a[0][0],max(a[1][1],a[2][2]));
    if scale<=0.0 { block_output.status.x=2u; return; }
    let an=a*(1.0/scale);
    let l00=sqrt(an[0][0]); let l10=an[0][1]/l00; let l20=an[0][2]/l00;
    let l11=sqrt(an[1][1]-l10*l10); let l21=(an[1][2]-l20*l10)/l11;
    let last=an[2][2]-l20*l20-l21*l21;
    if !(l00>0.0 && l11>0.0 && last>0.0) || !finite(last) { block_output.status.x=2u; return; }
    axes[0]=normalize(input.normal_gap.xyz);
    let absn=abs(axes[0]); var axis=vec3<f32>(1.0,0.0,0.0);
    if absn.y<absn.x && absn.y<=absn.z { axis=vec3<f32>(0.0,1.0,0.0); }
    else if absn.z<absn.x && absn.z<absn.y { axis=vec3<f32>(0.0,0.0,1.0); }
    axes[1]=normalize(cross(axis,axes[0])); axes[2]=cross(axes[0],axes[1]);
    for (var i=0u;i<3u;i++) {
        let flags=(mask>>(2u*i))&3u;
        if flags==3u { return; }
        axes[3u+i]=vec3<f32>(0.0);
        if flags!=0u { axes[3u+i][i]=select(1.0,-1.0,flags==2u); }
    }
    mu=input.machine_mu.w;
    point_inverse=a; free_soil=input.soil_mass.xyz; free_machine=input.machine_mu.xyz;
    soil_mass=input.soil_mass.w; contact_gap=input.normal_gap.w;
    for (var i=0u;i<6u;i++) {
        b[i]=dot(axes[i],input.soil_mass.xyz);
        if i<3u { b[i]-=dot(axes[i],input.machine_mu.xyz); }
        if i==0u { b[i]+=input.normal_gap.w; }
        for (var j=0u;j<6u;j++) {
            h[i][j]=dot(axes[i],axes[j])/input.soil_mass.w;
            if i<3u && j<3u { h[i][j]+=dot(axes[i],a*axes[j]); }
        }
        rho[i]=1.0;
        if h[i][i]>0.0 { rho[i]=1.0/h[i][i]; }
    }
    rho[1]=1.0/max(h[1][1],h[2][2]); rho[2]=rho[1];
    var x:array<f32,6>; var tail:array<f32,6>;
    var restarted=false;
    for (var iteration=0u;iteration<iterations;iteration++) {
        block_output.status.y=iteration+1u;
        let f=residual(x,tail,true); let old_merit=merit(f);
        if !finite(old_merit) { block_output.status.x=4u; return; }
        if old_merit==0.0 { publish(input,x,tail,a,0.0); return; }
        let delta=direction(f);
        if !linear_ok { block_output.status.x=64u; return; }
        if old_merit<=1e-12 && small_physical_correction(x,delta) { publish(input,x,tail,a,sqrt(old_merit)); if block_output.status.x!=32u { return; } }
        var step=1.0; var accepted=false;
        for (var backtrack=0u;backtrack<32u;backtrack++) {
            var trial:array<f32,6>; var trial_tail:array<f32,6>;
            for (var i=0u;i<6u;i++) {
                let sum=two_sum(x[i],step*delta[i]); let next=two_sum(sum.x,sum.y+tail[i]);
                trial[i]=next.x; trial_tail[i]=next.y;
            }
            let candidate=merit(residual(trial,trial_tail,false));
            if finite(candidate) && candidate<old_merit && candidate<=(1.0-2e-4*step)*old_merit { x=trial; tail=trial_tail; accepted=true; break; }
            step*=0.5; block_output.status.z+=1u;
        }
        if !accepted {
            if !restarted && iteration+1u<iterations {
                restarted=true;
                for (var i=0u;i<6u;i++) { for (var j=0u;j<6u;j++) {
                    jac[i][j]=h[i][j];
                    if dot(axes[i],axes[i])==0.0 { jac[i][j]=select(0.0,1.0,i==j); }
                } }
                let sticking=direction(b);
                var admissible=linear_ok && norm(vec2<f32>(sticking[1],sticking[2]))<=mu*sticking[0];
                for (var i=0u;i<6u;i++) {
                    admissible=admissible && finite(sticking[i]);
                    if i==0u || i>=3u { admissible=admissible && sticking[i]>=0.0; }
                }
                if admissible {
                    x=sticking;
                    for (var i=0u;i<6u;i++) { tail[i]=0.0; }
                    continue;
                }
            }
            block_output.status.x=8u; return;
        }
    }
    block_output.status.x=16u;
}
