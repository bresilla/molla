var<workgroup> direct_quality_records:array<vec4<f32>,128>;
var<workgroup> direct_quality_count:u32;
var<workgroup> direct_quality_overflow:u32;

fn direct_contact_relative(c:u32)->f32 {
    var scale=1e-30;
    for(var i=0u;i<6u;i++) {
        let a=component(scratch[c],i);let b=blocks[c].residual[i];
        if !finite(a) || !finite(b) {return 3.402823e38;}
        scale=max(scale,max(abs(a),abs(b)));
    }
    var error=0.0;var rhs=0.0;
    for(var i=0u;i<6u;i++) {
        let a=component(scratch[c],i)/scale;let b=blocks[c].residual[i]/scale;
        error+=a*a;rhs+=b*b;
    }
    if rhs==0.0 {return select(0.0,3.402823e38,error>0.0);}
    let ratio=sqrt(error)/sqrt(rhs);
    return select(3.402823e38,ratio,finite(ratio));
}

fn record_direct_quality(global:f32) {
    var worst=vec2<f32>(0.0,-1.0);
    for(var slot=lane;slot<status.active_count;slot+=256u) {
        let c=active_indices[slot];let ratio=direct_contact_relative(c);
        if worst.y<0.0 || ratio>worst.x || (ratio==worst.x && f32(c)<worst.y) {worst=vec2<f32>(ratio,f32(c));}
    }
    parts[lane]=worst;workgroupBarrier();
    if lane==0u {
        var best=vec2<f32>(0.0,-1.0);
        for(var i=0u;i<256u;i++) {
            let x=parts[i];
            if x.y>=0.0 && (best.y<0.0 || x.x>best.x || (x.x==best.x && x.y<best.y)) {best=x;}
        }
        if direct_quality_count<128u {direct_quality_records[direct_quality_count]=vec4<f32>(global,best,f32(report.steps));direct_quality_count++;}
        else {direct_quality_overflow=1u;}
    }
    workgroupBarrier();
}

fn export_direct_quality() {
    workgroupBarrier();storageBarrier();
    if lane==0u {condensed[0]=vec2<f32>(f32(direct_quality_count),f32(direct_quality_overflow));}
    for(var i=lane;i<128u;i+=256u) {
        let value=direct_quality_records[i];condensed[1u+2u*i]=value.xy;condensed[2u+2u*i]=value.zw;
    }
}

@compute @workgroup_size(64)
fn newton(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {direct_quality_count=0u;direct_quality_overflow=0u;}
    for(var i=lane;i<128u;i+=256u) {direct_quality_records[i]=vec4<f32>(0.0);}
    workgroupBarrier();quality_traced_newton(id);export_direct_quality();
}
