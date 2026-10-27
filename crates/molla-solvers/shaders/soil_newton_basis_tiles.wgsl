fn basis_address(index:u32,offset:u32)->vec2<u32> {
    let count=(params.iterations+1u)*max(params.contacts,1u);
    let first=(index/32u)*32u;let width=min(32u,count-first);
    return vec2<u32>(16u*first+offset*width+index%32u,width);
}
fn basis_load(index:u32,offset:u32)->Six {
    let a=basis_address(index,offset);let b=a.x;let s=a.y;
    return Six(vec4<f32>(basis_words[b],basis_words[b+s],basis_words[b+2u*s],basis_words[b+3u*s]),
        vec4<f32>(basis_words[b+4u*s],basis_words[b+5u*s],basis_words[b+6u*s],basis_words[b+7u*s]));
}
fn basis_store(index:u32,offset:u32,v:Six) {
    let a=basis_address(index,offset);let b=a.x;let s=a.y;
    basis_words[b]=v.k.x;basis_words[b+s]=v.k.y;basis_words[b+2u*s]=v.k.z;basis_words[b+3u*s]=v.k.w;
    basis_words[b+4u*s]=v.w.x;basis_words[b+5u*s]=v.w.y;basis_words[b+6u*s]=v.w.z;basis_words[b+7u*s]=v.w.w;
}
