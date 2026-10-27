struct PairVector { high: vec3<f32>, low: vec3<f32> }
struct PairSpatial { linear: PairVector, angular: PairVector }
@group(0) @binding(19) var<storage,read> input_velocity_low: array<Wrench>;
@group(0) @binding(20) var<storage,read_write> output_velocity_low: array<Wrench>;
@group(0) @binding(21) var<storage,read_write> output_rate_high: array<f32>;
var<workgroup> pair_scratch: array<atomic<u32>,1>;
var<private> pair_lane: u32;
fn pr(value: f32) -> f32 {
    atomicStore(&pair_scratch[pair_lane],bitcast<u32>(value));
    return bitcast<f32>(atomicAdd(&pair_scratch[pair_lane],0u));
}
fn ps(a: f32,b: f32) -> vec2<f32> {
    let s=pr(a+b); let vb=pr(s-a); let va=pr(s-vb);
    return vec2<f32>(s,pr(pr(a-va)+pr(b-vb)));
}
fn pa(a: vec2<f32>,b: vec2<f32>) -> vec2<f32> {
    let h=ps(a.x,b.x); let l=ps(a.y,b.y); let j=ps(h.x,pr(h.y+l.x));
    return ps(j.x,pr(j.y+l.y));
}
fn pm(a: vec2<f32>,b: vec2<f32>) -> vec2<f32> {
    let h=pr(a.x*b.x);
    return ps(h,pr(fma(a.x,b.x,-h)+pr(a.x*b.y)+pr(a.y*b.x)+pr(a.y*b.y)));
}
fn pv(v: vec3<f32>) -> PairVector { return PairVector(v,vec3<f32>(0.0)); }
fn pzero() -> PairSpatial { return PairSpatial(pv(vec3<f32>(0.0)),pv(vec3<f32>(0.0))); }
fn padd(a: PairVector,b: PairVector) -> PairVector {
    var result: PairVector;
    for (var i=0u;i<3u;i++) {let r=pa(vec2<f32>(a.high[i],a.low[i]),vec2<f32>(b.high[i],b.low[i])); result.high[i]=r.x; result.low[i]=r.y;}
    return result;
}
fn pneg(a: PairVector) -> PairVector { return PairVector(-a.high,-a.low); }
fn psub(a: PairVector,b: PairVector) -> PairVector { return padd(a,pneg(b)); }
fn pscale(a: PairVector,b: f32) -> PairVector {
    var result: PairVector;
    for (var i=0u;i<3u;i++) {let r=pm(vec2<f32>(a.high[i],a.low[i]),vec2<f32>(b,0.0)); result.high[i]=r.x; result.low[i]=r.y;}
    return result;
}
fn pcolumn(v: vec3<f32>,rate: vec2<f32>) -> PairVector {
    var result: PairVector;
    for (var i=0u;i<3u;i++) {let r=pm(vec2<f32>(v[i],0.0),rate); result.high[i]=r.x; result.low[i]=r.y;}
    return result;
}
fn pdivide(v: PairVector,d: f32) -> PairVector {
    let h=pv(v.high/d); let residual=psub(v,pscale(h,d));
    return padd(h,pv((residual.high+residual.low)/d));
}
fn pcross(a: PairVector,b: PairVector) -> PairVector {
    var result: PairVector;
    for (var i=0u;i<3u;i++) {
        let r=(i+1u)%3u; let c=(i+2u)%3u;
        let p=pm(vec2<f32>(a.high[r],a.low[r]),vec2<f32>(b.high[c],b.low[c]));
        let n=pm(vec2<f32>(a.high[c],a.low[c]),vec2<f32>(b.high[r],b.low[r]));
        let value=pa(p,-n); result.high[i]=value.x; result.low[i]=value.y;
    }
    return result;
}
fn pmat(m: mat3x3<f32>,v: PairVector) -> PairVector {
    var result=pv(vec3<f32>(0.0));
    for (var i=0u;i<3u;i++) {result=padd(result,pcolumn(m[i],vec2<f32>(v.high[i],v.low[i])));}
    return result;
}
fn protate(q: vec4<f32>,v: PairVector) -> PairVector {
    let xyz=pv(q.xyz);
    return padd(v,pscale(pcross(xyz,padd(pcross(xyz,v),pscale(v,q.w))),2.0));
}
fn pinertia(mass: f32,c: vec3<f32>,i: mat3x3<f32>,v: PairSpatial) -> PairSpatial {
    let p=pscale(padd(v.linear,pcross(v.angular,pv(c))),mass);
    return PairSpatial(p,padd(pmat(i,v.angular),pcross(pv(c),p)));
}
fn pfinite(v: PairVector) -> bool { return finite(v.high) && finite(v.low) && finite(v.high+v.low); }
fn psfinite(v: PairSpatial) -> bool { return pfinite(v.linear) && pfinite(v.angular); }
fn ppfinite(v: vec2<f32>) -> bool { return all(abs(v)<=vec2<f32>(3.402823e38)) && abs(v.x+v.y)<=3.402823e38; }
fn pbody(body: u32,output: bool) -> PairSpatial {
    if output {return PairSpatial(PairVector(output_velocity[body].linear.xyz,output_velocity_low[body].linear.xyz),PairVector(output_velocity[body].angular.xyz,output_velocity_low[body].angular.xyz));}
    return PairSpatial(PairVector(input_velocity[body].linear.xyz,input_velocity_low[body].linear.xyz),PairVector(input_velocity[body].angular.xyz,input_velocity_low[body].angular.xyz));
}
struct PairFactor { vectors: mat3x3<f32>, diagonal: vec3<f32>, scale: f32 }
fn pfactor(matrix: mat3x3<f32>) -> PairFactor {
    let scale=max(max(length(matrix[0]),length(matrix[1])),length(matrix[2]));
    if scale==0.0 {return PairFactor(identity(),vec3<f32>(0.0),1.0);}
    var a=matrix*(1.0/scale); var vectors=identity();
    for (var sweep=0u;sweep<24u;sweep++) {
        var p=0u;var q=1u;
        if abs(a[0][2])>abs(a[p][q]) {p=0u;q=2u;}
        if abs(a[1][2])>abs(a[p][q]) {p=1u;q=2u;}
        if abs(a[p][q])<=1e-7 {break;}
        let angle=0.5*atan2(2.0*a[p][q],a[q][q]-a[p][p]);
        let c=cos(angle);let s=sin(angle);var r=identity();
        r[p][p]=c;r[q][q]=c;r[p][q]= -s;r[q][p]=s;
        a=transpose(r)*a*r;vectors=vectors*r;
    }
    return PairFactor(vectors,vec3<f32>(a[0][0],a[1][1],a[2][2]),scale);
}
fn psolve(factor: PairFactor,rhs: PairVector) -> PairVector {
    let projected=pmat(transpose(factor.vectors),rhs);
    var answer=pv(vec3<f32>(0.0));
    for (var k=0u;k<3u;k++) {
        if factor.diagonal[k]>1e-7 {
            let d=pr(factor.diagonal[k]*factor.scale);
            let v=vec2<f32>(projected.high[k],projected.low[k]);
            let high=pr(v.x/d); let residual=pa(v,-pm(vec2<f32>(high,0.0),vec2<f32>(d,0.0)));
            let result=pa(vec2<f32>(high,0.0),vec2<f32>(pr(pr(residual.x+residual.y)/d),0.0));
            answer=padd(answer,pcolumn(factor.vectors[k],result));
        }
    }
    return answer;
}

@compute @workgroup_size(1)
fn transport(@builtin(global_invocation_id) gid: vec3<u32>,@builtin(local_invocation_index) lane: u32) {
    pair_lane=lane;
    let art=gid.x;
    if art>=params.count || atomicLoad(&status)!=0u {return;}
    let range=articulations[art];
    if range.z==0u || active_mask[art]==0u {return;}
    let root=joints[range.x]; let origin=root.child;
    for (var index=range.x;index<range.y;index++) {
        let j=joints[index];
        if !psfinite(pbody(j.child,false)) || !psfinite(pbody(j.child,true)) {atomicOr(&status,1u);return;}
        for (var d=0u;d<width(j.kind);d++) {
            if !ppfinite(input_rate[j.dof+d]) || !ppfinite(output_rate[j.dof+d]) || !ppfinite(delta[j.dof+d]) {atomicOr(&status,1u);return;}
            if params.correction!=0u && !ppfinite(response_delta[j.dof+d]) {atomicOr(&status,1u);return;}
        }
    }
    var velocity: array<PairSpatial,__ART_WIDTH__>;
    var acceleration: array<PairSpatial,__ART_WIDTH__>;
    var corrected: array<PairSpatial,__ART_WIDTH__>;
    var projected: array<PairSpatial,__ART_WIDTH__>;
    var desired=pzero(); var output=pzero();
    var reference_offset=vec3<f32>(0.0); var output_offset=vec3<f32>(0.0); var total_mass=0.0;
    let output_anchor=center(root,origin,2u);
    for (var index=range.x;index<range.y;index++) {
        let k=index-range.x; let j=joints[index];
        let f=frame(j,range.x,origin,0u); let fp=frame(j,range.x,origin,1u);
        var vj=pzero(); var aj=pzero(); var vc=pzero(); var vr=pzero();
        for (var d=0u;d<width(j.kind);d++) {
            let s=column(j,d,f); let sp=column(j,d,fp);
            let rate=input_rate[j.dof+d]; let next=output_rate[j.dof+d];
            vj.linear=padd(vj.linear,pcolumn(s.linear,rate)); vj.angular=padd(vj.angular,pcolumn(s.angular,rate));
            var change=delta[j.dof+d];
            if params.correction!=0u {change=pa(change,response_delta[j.dof+d]);}
            aj.linear=padd(aj.linear,pcolumn(s.linear,change)); aj.angular=padd(aj.angular,pcolumn(s.angular,change));
            vc.linear=padd(vc.linear,pcolumn(s.linear,next)); vc.angular=padd(vc.angular,pcolumn(s.angular,next));
            vr.linear=padd(vr.linear,pcolumn(sp.linear,next)); vr.angular=padd(vr.angular,pcolumn(sp.angular,next));
        }
        var vi=vj; var ai=aj;
        if j.parent>=0 {
            let p=u32(j.parent);
            vi.linear=padd(vi.linear,velocity[p].linear); vi.angular=padd(vi.angular,velocity[p].angular);
            ai.linear=padd(ai.linear,acceleration[p].linear); ai.angular=padd(ai.angular,acceleration[p].angular);
            vc.linear=padd(vc.linear,corrected[p].linear); vc.angular=padd(vc.angular,corrected[p].angular);
            vr.linear=padd(vr.linear,projected[p].linear); vr.angular=padd(vr.angular,projected[p].angular);
        }
        ai.linear=padd(ai.linear,pscale(padd(pcross(vi.angular,vj.linear),pcross(vi.linear,vj.angular)),params.dt));
        ai.angular=padd(ai.angular,pscale(pcross(vi.angular,vj.angular),params.dt));
        if j.kind>=4u && j.kind<=6u {
            let translation=protate(f.rotation,PairVector(vec3<f32>(input_rate[j.dof].x,input_rate[j.dof+1u].x,input_rate[j.dof+2u].x),vec3<f32>(input_rate[j.dof].y,input_rate[j.dof+1u].y,input_rate[j.dof+2u].y)));
            ai.linear=psub(ai.linear,pscale(pcross(vj.angular,translation),params.dt));
        }
        velocity[k]=vi; acceleration[k]=ai; corrected[k]=vc; projected[k]=vr;
        let c=center(j,origin,0u); let cp=center(j,origin,1u); let co=center(j,origin,2u);
        let ii=inertia(j,0u); let ip=inertia(j,1u); let io=inertia(j,2u);
        let mv=pinertia(j.mass.x,c,ii,vi); let ma=pinertia(j.mass.x,c,ii,ai);
        let mc=pinertia(j.mass.x,c,ii,vc); let mr=pinertia(j.mass.x,cp,ip,vr);
        let initial=pbody(j.child,false); let final_velocity=pbody(j.child,true);
        let p0=pscale(initial.linear,j.mass.x); let pn=pscale(final_velocity.linear,j.mass.x);
        desired.linear=padd(desired.linear,padd(padd(p0,ma.linear),padd(pscale(pcross(vi.angular,mv.linear),params.dt),psub(mr.linear,mc.linear))));
        desired.angular=padd(desired.angular,padd(padd(pmat(ii,initial.angular),pcross(pv(c),p0)),
            padd(ma.angular,padd(pscale(padd(pcross(vi.angular,mv.angular),pcross(vi.linear,mv.linear)),params.dt),psub(mr.angular,mc.angular)))));
        output.linear=padd(output.linear,pn); output.angular=padd(output.angular,padd(pmat(io,final_velocity.angular),pcross(pv(co),pn)));
        total_mass+=j.mass.x;
        reference_offset+=center_delta(j,1u,root,1u)*j.mass.x; output_offset+=center_delta(j,2u,root,2u)*j.mass.x;
    }
    if !(total_mass>0.0 && total_mass<=3.402823e38) || !psfinite(desired) || !psfinite(output) {atomicOr(&status,1u);return;}
    reference_offset/=total_mass; output_offset/=total_mass;
    let output_center=output_anchor+output_offset;
    let shift=padd(pv(center_delta(root,1u,root,2u)+(reference_offset-output_offset)),pscale(pdivide(desired.linear,total_mass),params.dt));
    let linear_delta=psub(desired.linear,output.linear);
    let angular_delta=psub(psub(psub(desired.angular,output.angular),pcross(shift,output.linear)),pcross(padd(pv(output_center),shift),linear_delta));
    var composite=mat3x3<f32>();
    for (var index=range.x;index<range.y;index++) {
        let j=joints[index]; let r=center_delta(j,2u,root,2u)-output_offset;
        composite+=inertia(j,2u)+j.mass.x*(identity()*dot(r,r)-mat3x3<f32>(r*r.x,r*r.y,r*r.z));
    }
    let factor=pfactor(composite);
    var spin=psolve(factor,angular_delta);
    for (var iteration=0u;iteration<2u;iteration++) {
        let residual=psub(angular_delta,pmat(composite,spin));
        spin=padd(spin,psolve(factor,residual));
    }
    let residual=psub(pmat(composite,spin),angular_delta);
    let scale=length(desired.angular.high+desired.angular.low)+length(output.angular.high+output.angular.low)
        +length(output_center+shift.high+shift.low)*(length(desired.linear.high+desired.linear.low)+length(output.linear.high+output.linear.low));
    if !pfinite(shift) || !pfinite(spin) || !pfinite(residual) || length(residual.high+residual.low)>3e-5*scale {atomicOr(&status,1u);return;}
    let f=frame(root,range.x,origin,2u); let inverse=vec4<f32>(-f.rotation.xyz,f.rotation.w);
    let direction=select(1.0,-1.0,root.reversed!=0u);
    let translation=padd(pdivide(linear_delta,total_mass),pcross(spin,pv(f.position.xyz-output_center)));
    let root_linear=pscale(protate(inverse,translation),direction); let root_angular=pscale(protate(inverse,spin),direction);
    let coordinate_shift=pscale(protate(inverse,shift),direction);
    var rates: array<vec2<f32>,6>; var coordinates: array<vec2<f32>,3>;
    for (var axis=0u;axis<3u;axis++) {
        rates[axis]=pa(output_rate[root.dof+axis],vec2<f32>(root_linear.high[axis],root_linear.low[axis]));
        rates[axis+3u]=pa(output_rate[root.dof+3u+axis],vec2<f32>(root_angular.high[axis],root_angular.low[axis]));
        let c=root.coordinate+axis;
        coordinates[axis]=pa(vec2<f32>(output_coordinate[c],bitcast<f32>(atomicLoad(&coordinate_low[c]))),vec2<f32>(coordinate_shift.high[axis],coordinate_shift.low[axis]));
        if !ppfinite(rates[axis]) || !ppfinite(rates[axis+3u]) || !ppfinite(coordinates[axis]) {atomicOr(&status,1u);return;}
    }
    var next_velocity: array<PairSpatial,__ART_WIDTH__>; var next_position: array<PairVector,__ART_WIDTH__>;
    for (var index=range.x;index<range.y;index++) {
        let j=joints[index]; let k=index-range.x; let c=center(j,origin,2u); let before=pbody(j.child,true);
        next_velocity[k]=PairSpatial(padd(before.linear,padd(pdivide(linear_delta,total_mass),pcross(spin,pv(c-output_center)))),padd(before.angular,spin));
        next_position[k]=padd(PairVector(output_pose[j.child].position.xyz,low(j.child,2u)),shift);
        if !psfinite(next_velocity[k]) || !pfinite(next_position[k]) {atomicOr(&status,1u);return;}
    }
    for (var axis=0u;axis<3u;axis++) {
        output_rate[root.dof+axis]=rates[axis]; output_rate[root.dof+3u+axis]=rates[axis+3u];
        output_rate_high[root.dof+axis]=rates[axis].x; output_rate_high[root.dof+3u+axis]=rates[axis+3u].x;
        let c=root.coordinate+axis; output_coordinate[c]=coordinates[axis].x; atomicStore(&coordinate_low[c],bitcast<u32>(coordinates[axis].y));
    }
    for (var index=range.x;index<range.y;index++) {
        let j=joints[index]; let k=index-range.x; let v=next_velocity[k];
        output_velocity[j.child]=Wrench(vec4<f32>(v.linear.high,0.0),vec4<f32>(v.angular.high,0.0));
        output_velocity_low[j.child]=Wrench(vec4<f32>(v.linear.low,0.0),vec4<f32>(v.angular.low,0.0));
        output_pose[j.child].position=vec4<f32>(next_position[k].high,0.0);
        for (var axis=0u;axis<3u;axis++) {atomicStore(&output_low[4u*j.child+axis],bitcast<u32>(next_position[k].low[axis]));}
    }
}
