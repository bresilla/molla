fn soa_residual(c:u32,i:u32)->f32 {return block_words[i*max(params.contacts,1u)+c];}
fn soa_set_residual(c:u32,i:u32,v:f32) {block_words[i*max(params.contacts,1u)+c]=v;}
fn soa_direct(c:u32,i:u32)->vec2<f32> {
    let n=max(params.contacts,1u);let at=6u*n+2u*(i*n+c);
    return vec2<f32>(block_words[at],block_words[at+1u]);
}
fn soa_set_direct(c:u32,i:u32,v:vec2<f32>) {
    let n=max(params.contacts,1u);let at=6u*n+2u*(i*n+c);
    block_words[at]=v.x;block_words[at+1u]=v.y;
}
fn soa_machine(c:u32,i:u32)->vec2<f32> {
    let n=max(params.contacts,1u);let at=78u*n+2u*(i*n+c);
    return vec2<f32>(block_words[at],block_words[at+1u]);
}
fn soa_set_machine(c:u32,i:u32,v:vec2<f32>) {
    let n=max(params.contacts,1u);let at=78u*n+2u*(i*n+c);
    block_words[at]=v.x;block_words[at+1u]=v.y;
}
fn soa_inverse(c:u32,i:u32)->f32 {return inverse_words[i*max(params.contacts,1u)+c];}
fn soa_set_inverse(c:u32,i:u32,v:f32) {inverse_words[i*max(params.contacts,1u)+c]=v;}
fn soa_clear_block(c:u32) {
    for(var i=0u;i<6u;i++) {soa_set_residual(c,i,0.0);}
    for(var i=0u;i<36u;i++) {soa_set_direct(c,i,vec2<f32>(0.0));}
    for(var i=0u;i<18u;i++) {soa_set_machine(c,i,vec2<f32>(0.0));}
}
fn soa_clear_inverse(c:u32) {for(var i=0u;i<36u;i++) {soa_set_inverse(c,i,0.0);}}
fn soa_store_residual(c:u32,v:array<f32,6>) {for(var i=0u;i<6u;i++) {soa_set_residual(c,i,v[i]);}}
fn soa_load_residual(c:u32)->array<f32,6> {
    var v:array<f32,6>;for(var i=0u;i<6u;i++) {v[i]=soa_residual(c,i);}return v;
}
