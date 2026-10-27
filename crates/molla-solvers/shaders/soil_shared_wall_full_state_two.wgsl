fn coop_coefficient(start:array<vec2<f32>,32>,has_previous:bool)->f32 {
    let n=shared_params.width;
    var current:array<f32,32>; var d1:array<f32,32>; var d2:array<f32,32>;
    let second=shared_status.sweeps>2u;
    for (var i=0u;i<n;i++) {
        let delta=two_sum(shared_q[i].x,-start[i].x);
        current[i]=rounded(delta.x+delta.y+shared_q[i].y-start[i].y);
        d1[i]=current[i]-shared_previous_f[i];
        if second { d2[i]=shared_previous_f[i]-shared_status.full_older_q[i]; }
    }
    var a=vec2<f32>(0.0); var b=vec2<f32>(0.0); var cross=vec2<f32>(0.0);
    var r1=vec2<f32>(0.0); var r2=vec2<f32>(0.0);
    for (var i=0u;i<n;i++) { for (var j=0u;j<n;j++) {
        let m=shared_mass[i*n+j];
        a=add_product(a,d1[i]*m,d1[j],0.0);
        b=add_product(b,d2[i]*m,d2[j],0.0);
        cross=add_product(cross,d1[i]*m,d2[j],0.0);
        r1=add_product(r1,d1[i]*m,current[j],0.0);
        r2=add_product(r2,d2[i]*m,current[j],0.0);
    } }
    for (var slot=0u;slot<shared_status.control.y;slot++) {
        let c=shared_history[slot].active_index; let mass=shared_contacts[c].soil_mass.w;
        let p=shared_outputs[c].machine_impulse_normal.xyz; let w=shared_outputs[c].wall_impulse.xyz;
        var fp=vec3<f32>(0.0); var fw=vec3<f32>(0.0);
        for (var axis=0u;axis<3u;axis++) {
            let dp=two_sum(p[axis],-shared_history[c].full_start_p[axis]);
            let dw=two_sum(w[axis],-shared_history[c].full_start_w[axis]);
            fp[axis]=rounded(dp.x+dp.y); fw[axis]=rounded(dw.x+dw.y);
            if has_previous {
                let x=vec2<f32>(fp[axis]-shared_history[c].full_previous_p[axis],fw[axis]-shared_history[c].full_previous_w[axis]);
                var y=vec2<f32>(0.0);
                if second { y=vec2<f32>(shared_history[c].full_previous_p[axis]-shared_history[c].full_older_fp[axis],shared_history[c].full_previous_w[axis]-shared_history[c].full_older_fw[axis]); }
                let f=vec2<f32>(fp[axis],fw[axis]);
                for (var k=0u;k<2u;k++) {
                    a=add_product(a,x[k]/mass,x[k],0.0); b=add_product(b,y[k]/mass,y[k],0.0);
                    cross=add_product(cross,x[k]/mass,y[k],0.0);
                    r1=add_product(r1,x[k]/mass,f[k],0.0); r2=add_product(r2,y[k]/mass,f[k],0.0);
                }
            }
        }
        shared_history[c].full_older_fp=shared_history[c].full_previous_p;
        shared_history[c].full_older_fw=shared_history[c].full_previous_w;
        shared_history[c].full_previous_p=vec4<f32>(fp,0.0);
        shared_history[c].full_previous_w=vec4<f32>(fw,0.0);
    }
    let aa=a.x+a.y; let bb=b.x+b.y; let ab=cross.x+cross.y;
    let x=r1.x+r1.y; let y=r2.x+r2.y;
    var coefficient=vec2<f32>(0.0);
    if has_previous && finite(aa) && finite(x) && aa>1e-30 { coefficient.x=clamp(-x/aa,-2.0,2.0); }
    let scale=max(aa,bb);
    if second && finite(scale) && scale>1e-30 && finite(ab) && finite(x) && finite(y) {
        let an=aa/scale; let bn=bb/scale; let cn=ab/scale;
        let determinant=fma(an,bn,-cn*cn);
        if determinant>7.6293945e-6*an*bn {
            let xn=x/scale; let yn=y/scale;
            let trial=vec2<f32>(fma(cn,yn,-bn*xn),fma(cn,xn,-an*yn))/determinant;
            if all(abs(trial)<=vec2<f32>(3.402823e38)) {
                coefficient=trial*min(1.0,2.0/max(abs(trial.x)+abs(trial.y),1e-30));
            }
        }
    }
    if !all(abs(coefficient)<=vec2<f32>(3.402823e38)) { coefficient=vec2<f32>(0.0); }
    shared_status.control.w=bitcast<u32>(coefficient.y);
    for (var i=0u;i<n;i++) { shared_status.full_older_q[i]=shared_previous_f[i]; }
    shared_previous_f=current;
    return coefficient.x;
}
