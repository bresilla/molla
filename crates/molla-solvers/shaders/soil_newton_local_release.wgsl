struct ReleaseRoots { values:array<f32,8>, count:u32 }
struct ReleaseCandidates { values:array<vec3<f32>,20>, count:u32 }
struct ReleaseChoice { state:Six, score:f32, valid:u32 }
var<workgroup> release_queue:array<u32,8>;
var<workgroup> release_visited:array<u32,8>;
var<workgroup> release_choices:array<ReleaseChoice,8>;
var<workgroup> release_best:ReleaseChoice;
var<workgroup> release_contact:u32;

fn release_poly(p:array<f32,5>,degree:u32,x:f32)->f32 {
    var value=p[degree];var i=degree;
    loop {if i==0u {break;}i--;value=value*x+p[i];}
    return value;
}
fn release_root_add(roots:ptr<function,ReleaseRoots>,x:f32) {
    if !finite(x) || abs(x)>1.0 || (*roots).count==8u {return;}
    var position=0u;
    loop {
        if position==(*roots).count {break;}
        if abs((*roots).values[position]-x)<2e-6 {return;}
        if (*roots).values[position]>x {break;}position++;
    }
    var i=(*roots).count;
    loop {if i==position {break;}(*roots).values[i]=(*roots).values[i-1u];i--;}
    (*roots).values[position]=x;(*roots).count++;
}
fn release_unit_roots(input:array<f32,5>)->ReleaseRoots {
    var degree=4u;
    loop {if degree==0u || input[degree]!=0.0 {break;}degree--;}
    var previous=ReleaseRoots();
    for(var d=1u;d<=degree;d++) {
        var p:array<f32,5>;var scale=0.0;
        for(var i=0u;i<=d;i++) {
            var v=input[i+degree-d];
            for(var j=1u;j<=degree-d;j++) {v*=f32(i+j);}
            p[i]=v;scale=max(scale,abs(v));
        }
        if !finite(scale) || scale==0.0 {return ReleaseRoots();}
        for(var i=0u;i<=d;i++) {p[i]/=scale;}
        var result=ReleaseRoots();
        if d==1u {release_root_add(&result,-p[0]/p[1]);}
        else {
            var points:array<f32,10>;points[0]=-1.0;
            for(var i=0u;i<previous.count;i++) {points[i+1u]=previous.values[i];}
            points[previous.count+1u]=1.0;
            for(var i=0u;i<previous.count+2u;i++) {
                if abs(release_poly(p,d,points[i]))<=1e-6 {release_root_add(&result,points[i]);}
            }
            for(var i=0u;i<previous.count+1u;i++) {
                var lo=points[i];var hi=points[i+1u];let sign=release_poly(p,d,lo)>=0.0;
                if sign==(release_poly(p,d,hi)>=0.0) {continue;}
                for(var iteration=0u;iteration<64u;iteration++) {
                    let mid=0.5*(lo+hi);
                    if (release_poly(p,d,mid)>=0.0)==sign {lo=mid;}else {hi=mid;}
                }
                release_root_add(&result,0.5*(lo+hi));
            }
        }
        previous=result;
    }
    return previous;
}
fn release_merit(chosen:u32,next:Six)->f32 {
    var q:array<vec2<f32>,32>;let old=states[chosen];
    for(var i=0u;i<params.width;i++) {
        q[i]=rates[i];
        for(var axis=0u;axis<3u;axis++) {
            let delta=rate_add(vec2<f32>(next.k[axis],0.0),-vec2<f32>(old.high.k[axis],old.low.k[axis]));
            q[i]=rate_add(q[i],rate_mul(-response[chosen*params.width+i][axis],delta));
        }
    }
    var square=vec2<f32>(0.0);
    for(var slot=0u;slot<status.active_count;slot++) {
        let c=active_indices[slot];let node=contacts[c];var x=states[c];
        if c==chosen {x=State(next,Six());}
        var a:Local;a.mass=node.soil_mass.w;a.mu=node.friction.x;a.gap=node.normal_gap.w;a.wall=flags(c);
        let normal_length=fischer_length(node.normal_gap.xyz);var trace=vec2<f32>(0.0);var separation=vec2<f32>(0.0);
        for(var axis=0u;axis<3u;axis++) {
            a.paired_normal[axis]=fischer_div(vec2<f32>(node.normal_gap[axis],0.0),normal_length);
            a.normal[axis]=value(a.paired_normal[axis]);
            a.paired_k[axis]=vec2<f32>(x.high.k[axis],x.low.k[axis]);a.k[axis]=value(a.paired_k[axis]);
            a.paired_w[axis]=vec2<f32>(x.high.w[axis],x.low.w[axis]);a.w[axis]=value(a.paired_w[axis]);
            let soil=rate_add(rate_mul(a.mass,vec2<f32>(node.soil_mass[axis],0.0)),rate_add(a.paired_k[axis],a.paired_w[axis]));
            a.paired_soil[axis]=fischer_div(soil,vec2<f32>(a.mass,0.0));a.soil[axis]=value(a.paired_soil[axis]);
            var machine=vec2<f32>(0.0);
            for(var i=0u;i<params.width;i++) {
                machine=rate_add(machine,rate_mul(rows[c*params.width+i][axis],q[i]));
                trace=rate_add(trace,rate_mul(rows[c*params.width+i][axis],vec2<f32>(response[c*params.width+i][axis],0.0)));
            }
            a.paired_relative[axis]=rate_add(a.paired_soil[axis],-machine);a.relative[axis]=value(a.paired_relative[axis]);
            separation=rate_add(separation,rate_mul(node.normal_gap[axis],a.paired_relative[axis]));
        }
        a.paired_separation=rate_add(fischer_div(separation,normal_length),vec2<f32>(a.gap,0.0));
        a.paired_rho=fischer_div(vec2<f32>(1.0,0.0),rate_add(fischer_div(vec2<f32>(1.0,0.0),vec2<f32>(a.mass,0.0)),trace));
        let residual=fischer_residual(a);
        for(var j=0u;j<6u;j++) {square=rate_add(square,rate_mul(residual[j],vec2<f32>(residual[j],0.0)));}
    }
    let result=sqrt(max(value(square),0.0));return select(3.402823e38,result,finite(result));
}
fn release_consider(c:u32,mask:u32,k:vec3<f32>,other:vec3<f32>,machine:mat3x3<f32>,best:ptr<function,ReleaseChoice>) {
    let node=contacts[c];let n=normalize(node.normal_gap.xyz);let mass=node.soil_mass.w;let mu=node.friction.x;let wall=flags(c);
    let lambda=dot(n,k);var w=vec3<f32>(0.0);var releases=false;let old=states[c];
    if !all(abs(k)<=vec3<f32>(3.402823e38)) || lambda<0.0 || length(k-n*lambda)>mu*lambda+1e-5*(1.0+length(k)) {return;}
    for(var axis=0u;axis<3u;axis++) {
        if (mask&(1u<<axis))!=0u {w[axis]=-mass*node.soil_mass[axis]-k[axis];}
        releases=releases || (wall[axis]*(old.high.w[axis]+old.low.w[axis])>1e-6 && w[axis]==0.0);
    }
    if !releases || !all(abs(w)<=vec3<f32>(3.402823e38)) {return;}
    let soil=node.soil_mass.xyz+(k+w)/mass;
    if any(wall*w<vec3<f32>(0.0)) || any(wall*soil<vec3<f32>(-1e-5)) {return;}
    let relative=soil-other+machine*k;let separation=dot(n,relative)+node.normal_gap.w;
    let slip=relative-n*dot(n,relative);let tangent=k-n*lambda;
    let ps=1.0+length(k);let vs=1.0+length(relative)+node.normal_gap.w;
    if separation < -1e-5*vs || abs(lambda*separation)>1e-5*ps*vs {return;}
    if length(slip)>1e-6 && length(tangent+mu*lambda*normalize(slip))>1e-5*ps {return;}
    let next=Six(vec4<f32>(k,0.0),vec4<f32>(w,0.0));let score=release_merit(c,next);
    if score<(*best).score {(*best)=ReleaseChoice(next,score,1u);}
}
fn release_local(c:u32,mask:u32)->ReleaseChoice {
    var best=ReleaseChoice(Six(),3.402823e38,0u);let wall=flags(c);var bits=0u;
    for(var axis=0u;axis<3u;axis++) {if wall[axis]!=0.0 {bits|=1u<<axis;}}
    if (mask&(~bits))!=0u {return best;}
    let node=contacts[c];let n=normalize(node.normal_gap.xyz);let mu=node.friction.x;let old=states[c];
    var machine=mat3x3<f32>();var other=vec3<f32>(0.0);
    for(var axis=0u;axis<3u;axis++) {
        var velocity=vec2<f32>(0.0);
        for(var i=0u;i<params.width;i++) {
            var without=rates[i];
            for(var j=0u;j<3u;j++) {without=rate_add(without,rate_mul(response[c*params.width+i][j],vec2<f32>(old.high.k[j],old.low.k[j])));}
            velocity=rate_add(velocity,rate_mul(rows[c*params.width+i][axis],without));
        }
        other[axis]=value(velocity);
        for(var j=0u;j<3u;j++) {
            var sum=vec2<f32>(0.0);
            for(var i=0u;i<params.width;i++) {sum=rate_add(sum,rate_mul(rows[c*params.width+i][axis],vec2<f32>(response[c*params.width+i][j],0.0)));}
            machine[j][axis]=value(sum);
        }
    }
    var a=machine;var soil=node.soil_mass.xyz;
    for(var axis=0u;axis<3u;axis++) {
        if (mask&(1u<<axis))!=0u {soil[axis]=0.0;}else {a[axis][axis]+=1.0/node.soil_mass.w;}
    }
    let candidates=release_candidates(n,mu,node.normal_gap.w,a,soil-other);
    for(var i=0u;i<candidates.count;i++) {release_consider(c,mask,candidates.values[i],other,machine,&best);}
    return best;
}
fn release_candidates(n:vec3<f32>,mu:f32,gap:f32,a:mat3x3<f32>,b:vec3<f32>)->ReleaseCandidates {
    var result=ReleaseCandidates();let bias=dot(n,b)+gap;
    if bias>=0.0 {result.values[result.count]=vec3<f32>(0.0);result.count++;}
    let determinant=dot(a[0],cross(a[1],a[2]));
    if abs(determinant)>1e-20 {
        let rhs=-b-n*gap;
        let sticking=vec3<f32>(dot(rhs,cross(a[1],a[2])),dot(a[0],cross(rhs,a[2])),dot(a[0],cross(a[1],rhs)))/determinant;
        result.values[result.count]=sticking;result.count++;
    }
    if mu==0.0 {
        let denominator=dot(n,a*n);
        if denominator>0.0 {result.values[result.count]=(-bias/denominator)*n;result.count++;}return result;
    }
    let axis=select(vec3<f32>(0.0,1.0,0.0),vec3<f32>(1.0,0.0,0.0),abs(n.x)<0.8);
    let t1=normalize(cross(n,axis));let t2=cross(n,t1);
    let directions=array<vec3<f32>,3>(n-mu*t1,-2.0*mu*t2,n+mu*t1);
    let tangents=array<vec3<f32>,3>(t1,2.0*t2,-t1);var p:array<f32,5>;
    for(var i=0u;i<3u;i++) {let product=a*directions[i];let v=b*dot(n,product)-bias*product;
        for(var j=0u;j<3u;j++) {p[i+j]+=dot(n,cross(v,tangents[j]));}
    }
    for(var reciprocal=0u;reciprocal<2u;reciprocal++) {
        let roots=release_unit_roots(p);
        for(var root=0u;root<=roots.count;root++) {
            var t=-t1;
            if root<roots.count {
                var r=roots.values[root];if reciprocal==1u {if abs(r)<1e-7 {continue;}r=1.0/r;}
                let angle=2.0*atan(r);t=t1*cos(angle)+t2*sin(angle);
            }
            let d=n-mu*t;let denominator=dot(n,a*d);
            if denominator==0.0 {continue;}
            let k=(-bias/denominator)*d;let u=b+a*k;let slip=u-n*dot(n,u);
            if dot(slip,t)>0.0 && length(cross(slip,t))<=1e-5*(1.0+length(slip)) {result.values[result.count]=k;result.count++;}
        }
        p=array<f32,5>(p[4],p[3],p[2],p[1],p[0]);
    }
    return result;
}
fn release_sliding_wall()->bool {
    if workgroupUniformLoad(&outer.releases)>=8u {return false;}
    if lane==0u {
        var scores:array<f32,8>;
        for(var i=0u;i<8u;i++) {release_queue[i]=0xffffffffu;}
        for(var slot=0u;slot<status.active_count;slot++) {
            let c=active_indices[slot];var visited=false;
            for(var i=0u;i<outer.releases;i++) {visited=visited || release_visited[i]==c;}
            if visited {continue;}
            var score=0.0;for(var i=0u;i<6u;i++) {score+=blocks[c].residual[i]*blocks[c].residual[i];}
            for(var i=0u;i<8u;i++) {if score>scores[i] {var j=7u;loop {if j==i {break;}scores[j]=scores[j-1u];release_queue[j]=release_queue[j-1u];j--;}scores[i]=score;release_queue[i]=c;break;}}
        }
        release_best=ReleaseChoice(Six(),3.402823e38,0u);release_contact=0xffffffffu;
    }
    workgroupBarrier();
    for(var rank=0u;rank<8u;rank++) {
        let c=workgroupUniformLoad(&release_queue[rank]);if c==0xffffffffu {continue;}
        if lane<8u {release_choices[lane]=release_local(c,lane);}
        workgroupBarrier();
        if lane==0u {for(var i=0u;i<8u;i++) {let candidate=release_choices[i];if candidate.valid!=0u && candidate.score<release_best.score {release_best=candidate;release_contact=c;}}}
        workgroupBarrier();
    }
    let chosen=workgroupUniformLoad(&release_contact);
    if chosen==0xffffffffu {return false;}
    if lane==0u {states[chosen]=State(release_best.state,Six());release_visited[outer.releases]=chosen;outer.releases++;}
    storageBarrier();return true;
}
