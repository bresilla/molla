struct BoundaryScan { key: vec3<u32>, step: f32, violation: f32 }

fn boundary_normal(face: u32) -> vec3<f32> {
    var normal=vec3<f32>(0.0);
    normal[face/2u]=select(1.0,-1.0,(face&1u)!=0u);
    return normal;
}
fn boundary_row(key: vec3<u32>, col: u32) -> f32 {
    let point_velocity=column(key.x,0u,col)+cross(column(key.x,3u,col),nodes[key.y].arm_mass.xyz);
    return dot(boundary_normal(key.z),point_velocity);
}
fn sum_residue(a: vec3<f32>, b: vec3<f32>, sum: vec3<f32>) -> vec3<f32> {
    return select((b-sum)+a,(a-sum)+b,abs(a)>=abs(b));
}
fn raw_multiplier(hi: f32, lo: f32, norm: f32) -> vec2<f32> {
    if hi+lo<0.0 { return vec2<f32>(0.0); }
    let value=hi/norm;
    return vec2<f32>(value,(fma(-value,norm,hi)+lo)/norm);
}
fn boundary_wrenches(working: ptr<function,array<vec3<u32>,32>>,
    lambda: ptr<function,array<f32,32>>, low: ptr<function,array<f32,32>>,
    norms: ptr<function,array<f32,32>>, count: u32) {
    for (var b=0u;b<params.bodies;b++) {
        var linear=vec3<f32>(0.0); var linear_low=vec3<f32>(0.0);
        var angular=vec3<f32>(0.0); var angular_low=vec3<f32>(0.0);
        for (var k=0u;k<count;k++) {
            let key=(*working)[k];
            if key.x!=b { continue; }
            let multiplier=raw_multiplier((*lambda)[k],(*low)[k],(*norms)[k]);
            let normal=boundary_normal(key.z); let arm=nodes[key.y].arm_mass.xyz;
            let p=normal*multiplier.x; let p_low=normal*multiplier.y;
            let product_a=arm.yzx*p.zxy; let product_b=arm.zxy*p.yzx;
            let h=product_a-product_b;
            let h_low=cross(arm,p_low)+fma(arm.yzx,p.zxy,-product_a)-fma(arm.zxy,p.yzx,-product_b);
            let next_linear=linear+p; let next_angular=angular+h;
            linear_low+=sum_residue(linear,p,next_linear)+p_low;
            angular_low+=sum_residue(angular,h,next_angular)+h_low;
            linear=next_linear; angular=next_angular;
        }
        linear+=linear_low; angular+=angular_low;
        if !finite3(linear) || !finite3(angular) { atomicOr(&status,4u); return; }
        boundary_impulse[2u*b]=vec4<f32>(linear,0.0);
        boundary_impulse[2u*b+1u]=vec4<f32>(angular,0.0);
    }
}
fn independent_boundary(key: vec3<u32>, lower: ptr<function,array<f32,1024>>,
    basis: ptr<function,array<f32,1024>>, count: u32) -> bool {
    if count==0u { return true; }
    let n=params.width;
    var row: array<f32,32>; var norm=0.0;
    for (var i=0u;i<n;i++) {
        var value=boundary_row(key,i);
        for (var j=0u;j<i;j++) { value-=(*lower)[i*n+j]*row[j]; }
        row[i]=value/(*lower)[i*n+i]; norm+=row[i]*row[i];
    }
    for (var sweep=0u;sweep<2u;sweep++) {
        for (var k=0u;k<count;k++) {
            var value=0.0;
            for (var i=0u;i<n;i++) { value+=(*basis)[k*n+i]*row[i]; }
            for (var i=0u;i<n;i++) { row[i]-=value*(*basis)[k*n+i]; }
        }
    }
    var residual=0.0;
    for (var i=0u;i<n;i++) { residual+=row[i]*row[i]; }
    if !finite(norm) || !finite(residual) { atomicOr(&status,4u); return false; }
    return residual>1e-12*norm;
}
fn scan_boundaries(x: ptr<function,array<f32,32>>, direction: ptr<function,array<f32,32>>,
    working: ptr<function,array<vec3<u32>,32>>, count: u32,
    lower: ptr<function,array<f32,1024>>, basis: ptr<function,array<f32,1024>>) -> BoundaryScan {
    var result=BoundaryScan(vec3<u32>(0xffffffffu),1.0,0.0);
    for (var b=0u;b<params.bodies;b++) {
        let span=spans[b];
        for (var k=0u;k<span.y;k++) {
            let index=span.x+k;
            if nodes[index].arm_mass.w==0.0 { continue; }
            let mask=u32(nodes[index].momentum.w);
            for (var face=0u;face<6u;face++) {
                if (mask&(1u<<face))==0u { continue; }
                let key=vec3<u32>(b,index,face);
                var value=0.0; var speed=0.0; var value_scale=0.0; var speed_scale=0.0;
                for (var i=0u;i<params.width;i++) {
                    let a=boundary_row(key,i);
                    value+=a*(*x)[i]; value_scale+=abs(a*(*x)[i]);
                    speed+=a*(*direction)[i]; speed_scale+=abs(a*(*direction)[i]);
                }
                if !finite(value) || !finite(speed) || !finite(value_scale) || !finite(speed_scale) {
                    atomicOr(&status,4u); return result;
                }
                result.violation=max(result.violation,-value/max(1.0,value_scale));
                var selected=false;
                for (var i=0u;i<count;i++) { selected=selected || all((*working)[i]==key); }
                if !selected && speed < -1e-6*max(speed_scale,1e-30) {
                    let step=max(0.0,value)/(-speed);
                    if step<result.step && independent_boundary(key,lower,basis,count) {
                        result.step=step; result.key=key;
                    }
                }
            }
        }
    }
    return result;
}

@compute @workgroup_size(1)
fn boundary() {
    if atomicLoad(&status)!=0u { return; }
    var present=false;
    for (var b=0u;b<params.bodies;b++) {
        present=present || boundary_impulse[2u*b].w!=0.0;
        boundary_impulse[2u*b].w=0.0;
    }
    if !present { return; }
    let n=params.width;
    var working: array<vec3<u32>,32>;
    var x: array<f32,32>;
    var direction: array<f32,32>;
    var lower: array<f32,1024>;
    var q: array<f32,1024>;
    for (var i=0u;i<n;i++) { x[i]=rates[i]; }
    let initial=scan_boundaries(&x,&direction,&working,0u,&lower,&q);
    if atomicLoad(&status)!=0u || initial.violation<=1e-7 { return; }

    var r: array<f32,1024>;
    var norms: array<f32,32>;
    var unconstrained: array<f32,32>;
    var y: array<f32,32>;
    var dy: array<f32,32>;
    var lambda: array<f32,32>;
    var lambda_low: array<f32,32>;
    var count=0u;
    var converged=false;
    for (var i=0u;i<n;i++) {
        for (var j=0u;j<=i;j++) {
            var value=system[i*n+j];
            for (var k=0u;k<j;k++) { value-=lower[i*n+k]*lower[j*n+k]; }
            if j==i {
                if !finite(value) || value<=0.0 { atomicOr(&status,2u); return; }
                lower[i*n+i]=sqrt(value);
            } else { lower[i*n+j]=value/lower[j*n+j]; }
        }
        var value=rhs[i];
        for (var j=0u;j<i;j++) { value-=lower[i*n+j]*unconstrained[j]; }
        unconstrained[i]=value/lower[i*n+i];
    }
    for (var iteration=0u;iteration<256u;iteration++) {
        for (var k=0u;k<count;k++) {
            var length2=0.0;
            for (var i=0u;i<n;i++) {
                var value=boundary_row(working[k],i);
                for (var j=0u;j<i;j++) { value-=lower[i*n+j]*q[k*n+j]; }
                q[k*n+i]=value/lower[i*n+i]; length2+=q[k*n+i]*q[k*n+i];
                r[i*n+k]=0.0;
            }
            norms[k]=sqrt(length2);
            if !finite(norms[k]) || norms[k]<=1e-20 { atomicOr(&status,192u); return; }
            for (var i=0u;i<n;i++) { q[k*n+i]/=norms[k]; }
            for (var sweep=0u;sweep<2u;sweep++) {
                for (var j=0u;j<k;j++) {
                    var value=0.0;
                    for (var i=0u;i<n;i++) { value+=q[j*n+i]*q[k*n+i]; }
                    r[j*n+k]+=value;
                    for (var i=0u;i<n;i++) { q[k*n+i]-=value*q[j*n+i]; }
                }
            }
            length2=0.0;
            for (var i=0u;i<n;i++) { length2+=q[k*n+i]*q[k*n+i]; }
            let diagonal=sqrt(length2);
            if !finite(diagonal) || diagonal<=1e-6 { atomicOr(&status,192u); return; }
            r[k*n+k]=diagonal;
            for (var i=0u;i<n;i++) { q[k*n+i]/=diagonal; }
        }
        for (var i=0u;i<n;i++) { dy[i]=unconstrained[i]-y[i]; }
        for (var k=0u;k<count;k++) {
            var value=0.0;
            for (var i=0u;i<n;i++) { value+=q[k*n+i]*unconstrained[i]; }
            lambda[k]=-value;
            for (var i=0u;i<n;i++) { dy[i]-=value*q[k*n+i]; }
        }
        for (var sweep=0u;sweep<2u;sweep++) {
            for (var k=0u;k<count;k++) {
                var value=0.0;
                for (var i=0u;i<n;i++) { value+=q[k*n+i]*dy[i]; }
                for (var i=0u;i<n;i++) { dy[i]-=value*q[k*n+i]; }
            }
        }
        var length2=0.0; var scale=0.0;
        for (var i=0u;i<n;i++) { length2+=dy[i]*dy[i]; scale+=unconstrained[i]*unconstrained[i]; }
        if !finite(length2) || !finite(scale) { atomicOr(&status,4u); return; }
        if length2<=1e-12*max(1.0,scale) {
            var smallest=0.0; var remove=0xffffffffu; var largest=0.0;
            for (var k=0u;k<count;k++) {
                let i=count-1u-k; var value=lambda[i];
                for (var j=i+1u;j<count;j++) { value-=r[i*n+j]*lambda[j]; }
                lambda[i]=value/r[i*n+i]; largest=max(largest,abs(lambda[i]));
                if lambda[i]<smallest { smallest=lambda[i]; remove=i; }
            }
            if smallest < -1e-6*max(1.0,largest) {
                for (var i=remove;i+1u<count;i++) { working[i]=working[i+1u]; }
                count-=1u; continue;
            }
            converged=true; break;
        }
        for (var k=0u;k<n;k++) {
            let i=n-1u-k; var value=y[i]; var speed=dy[i];
            for (var j=i+1u;j<n;j++) {
                value-=lower[j*n+i]*x[j]; speed-=lower[j*n+i]*direction[j];
            }
            x[i]=value/lower[i*n+i]; direction[i]=speed/lower[i*n+i];
        }
        let hit=scan_boundaries(&x,&direction,&working,count,&lower,&q);
        if atomicLoad(&status)!=0u { return; }
        if hit.violation>2e-5 { atomicOr(&status,8u); return; }
        for (var i=0u;i<n;i++) { y[i]+=hit.step*dy[i]; }
        if hit.key.x!=0xffffffffu {
            if count==n { atomicOr(&status,192u); return; }
            working[count]=hit.key; count+=1u;
        }
    }
    if !converged { atomicOr(&status,320u); return; }
    for (var k=0u;k<n;k++) {
        let i=n-1u-k; var value=y[i];
        for (var j=i+1u;j<n;j++) { value-=lower[j*n+i]*x[j]; }
        x[i]=value/lower[i*n+i]; direction[i]=0.0;
        if !finite(x[i]) { atomicOr(&status,4u); return; }
    }
    let final_scan=scan_boundaries(&x,&direction,&working,count,&lower,&q);
    if atomicLoad(&status)!=0u { return; }
    if final_scan.violation>2e-5 { atomicOr(&status,8u); return; }
    for (var refinement=0u;refinement<3u;refinement++) {
        boundary_wrenches(&working,&lambda,&lambda_low,&norms,count);
        if atomicLoad(&status)!=0u { return; }
        for (var i=0u;i<n;i++) {
            var value=-rhs[i];
            for (var j=0u;j<n;j++) { value=fma(system[i*n+j],x[j],value); }
            for (var b=0u;b<params.bodies;b++) {
                let linear=column(b,0u,i); let angular=column(b,3u,i);
                for (var axis=0u;axis<3u;axis++) {
                    value=fma(-linear[axis],boundary_impulse[2u*b][axis],value);
                    value=fma(-angular[axis],boundary_impulse[2u*b+1u][axis],value);
                }
            }
            for (var j=0u;j<i;j++) { value-=lower[i*n+j]*direction[j]; }
            direction[i]=value/lower[i*n+i];
        }
        for (var k=0u;k<count;k++) {
            var value=0.0;
            for (var i=0u;i<n;i++) { value+=q[k*n+i]*direction[i]; }
            dy[k]=value;
        }
        for (var k=0u;k<count;k++) {
            let i=count-1u-k; var value=dy[i];
            for (var j=i+1u;j<count;j++) { value-=r[i*n+j]*dy[j]; }
            dy[i]=value/r[i*n+i];
            lambda_low[i]+=dy[i];
        }
    }
    for (var k=0u;k<count;k++) {
        if !finite(lambda[k]+lambda_low[k]) || lambda[k]+lambda_low[k]<-1e-6 { atomicOr(&status,576u); return; }
    }
    for (var i=0u;i<n;i++) {
        var value=-rhs[i]; var scale=abs(rhs[i]);
        for (var j=0u;j<n;j++) { let term=system[i*n+j]*x[j]; value+=term; scale+=abs(term); }
        for (var k=0u;k<count;k++) {
            let multiplier=raw_multiplier(lambda[k],lambda_low[k],norms[k]);
            let term=boundary_row(working[k],i)*(multiplier.x+multiplier.y);
            value-=term; scale+=abs(term);
        }
        if !finite(value) || !finite(scale) || abs(value)>2e-5*max(scale,1e-30) {
            atomicOr(&status,8u); return;
        }
    }
    for (var k=0u;k<count;k++) {
        let key=working[k]; var speed=0.0; var scale=0.0;
        for (var i=0u;i<n;i++) { let term=boundary_row(key,i)*x[i]; speed+=term; scale+=abs(term); }
        if abs(speed)>2e-5*max(1.0,scale) { atomicOr(&status,8u); return; }
    }
    boundary_wrenches(&working,&lambda,&lambda_low,&norms,count);
    if atomicLoad(&status)!=0u { return; }
    for (var i=0u;i<n;i++) {
        var value=-rhs[i]; var scale=abs(rhs[i]);
        for (var j=0u;j<n;j++) { let term=system[i*n+j]*x[j]; value+=term; scale+=abs(term); }
        for (var b=0u;b<params.bodies;b++) {
            let linear=column(b,0u,i)*boundary_impulse[2u*b].xyz;
            let angular=column(b,3u,i)*boundary_impulse[2u*b+1u].xyz;
            for (var axis=0u;axis<3u;axis++) {
                value-=linear[axis]+angular[axis]; scale+=abs(linear[axis])+abs(angular[axis]);
            }
        }
        if !finite(value) || !finite(scale) || abs(value)>2e-5*max(scale,1e-30) {
            atomicOr(&status,8u); return;
        }
    }
    for (var i=0u;i<n;i++) { rates[i]=x[i]; }
}
