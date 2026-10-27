fn release_separating_wall()->bool {
    if workgroupUniformLoad(&outer.releases)>=8u {return false;}
    var best=vec2<f32>(0.0);
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];let node=contacts[c];let x=states[c];let wall=flags(c);
        let normal=normalize(node.normal_gap.xyz);
        for (var axis=0u;axis<3u;axis++) {
            let w=vec2<f32>(x.high.w[axis],x.low.w[axis]);
            let total=rate_add(vec2<f32>(x.high.k[axis],x.low.k[axis]),w);
            let soil=value(rate_mul(1.0/node.soil_mass.w,rate_add(rate_mul(node.soil_mass.w,vec2<f32>(node.soil_mass[axis],0.0)),total)));
            if wall[axis]*value(w)<=0.0 || wall[axis]*soil<=0.0 {continue;}
            var k=x.high.k.xyz+x.low.k.xyz;k[axis]=value(total);
            let lambda=dot(normal,k);
            if lambda<0.0 || length(k-normal*lambda)>node.friction.x*lambda {continue;}
            let score=value(w)*soil;let index=f32(c*3u+axis+1u);
            if finite(score) && score>best.x {best=vec2<f32>(score,index);}
        }
    }
    parts[lane]=best;workgroupBarrier();
    if lane==0u {
        var chosen=vec2<f32>(0.0);
        for (var i=0u;i<64u;i++) {
            let p=parts[i];
            if p.x>chosen.x || (p.x>0.0 && p.x==chosen.x && p.y<chosen.y) {chosen=p;}
        }
        outer.accepted=u32(chosen.y);
        if outer.accepted!=0u {
            let index=outer.accepted-1u;let c=index/3u;let axis=index%3u;var x=states[c];
            let k=rate_add(vec2<f32>(x.high.k[axis],x.low.k[axis]),vec2<f32>(x.high.w[axis],x.low.w[axis]));
            x.high.k[axis]=k.x;x.low.k[axis]=k.y;x.high.w[axis]=0.0;x.low.w[axis]=0.0;
            states[c]=x;outer.releases++;
        }
    }
    storageBarrier();return workgroupUniformLoad(&outer.accepted)!=0u;
}
