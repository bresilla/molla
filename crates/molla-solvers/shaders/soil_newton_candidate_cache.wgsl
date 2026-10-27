fn release_cached_candidate(c:u32,mask:u32,index:u32)->ReleaseChoice {
    var best=ReleaseChoice(Six(),3.402823e38,0u);let base=80u*mask;
    if index>=u32(h[base]) {return best;}
    var other=vec3<f32>(0.0);var machine=mat3x3<f32>();var k=vec3<f32>(0.0);
    for(var a=0u;a<3u;a++) {
        other[a]=h[base+1u+a];k[a]=h[base+13u+3u*index+a];
        for(var j=0u;j<3u;j++) {machine[j][a]=h[base+4u+3u*j+a];}
    }
    release_consider(c,mask,k,other,machine,&best);
    return best;
}
