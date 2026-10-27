struct Six { k:vec4<f32>, w:vec4<f32> }
struct State { high:Six, low:Six }
struct Query { position_mass:vec4<f32>, velocity_dt:vec4<f32>, body_boundary:vec4<u32> }
struct Contact { soil_mass:vec4<f32>, normal_gap:vec4<f32>, friction:vec4<f32>, boundary:vec4<u32> }
struct Entry { key:vec4<u32>, normal_mu:vec4<f32>, mass_dt:vec4<f32>, state:State }
struct Header { ready:u32, invalid:atomic<u32>, hits:atomic<u32>, nonzero:atomic<u32> }
struct Params { capacity:u32, nodes:u32, pad0:u32, pad1:u32 }
@group(0) @binding(0) var<storage,read> ids:array<u32>;
@group(0) @binding(1) var<storage,read> queries:array<Query>;
@group(0) @binding(2) var<storage,read> contacts:array<Contact>;
@group(0) @binding(3) var<storage,read_write> states:array<State>;
@group(0) @binding(4) var<storage,read> common_status:array<u32>;
@group(0) @binding(5) var<storage,read> solve_status:array<u32>;
@group(0) @binding(6) var<storage,read> publication_status:array<u32>;
@group(0) @binding(7) var<storage,read_write> history:array<Entry>;
@group(0) @binding(8) var<storage,read_write> header:Header;
@group(0) @binding(9) var<uniform> params:Params;
@group(0) @binding(10) var<storage,read_write> seeds:array<State>;
fn valid_six(x:Six)->bool {
    return all(abs(x.k)<=vec4<f32>(3.402823e38)) && all(abs(x.w)<=vec4<f32>(3.402823e38)) && x.k.w==0.0 && x.w.w==0.0;
}
@compute @workgroup_size(64)
fn restore(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x;if c>=params.capacity {return;}
    states[c]=State();if common_status[0]!=0u {return;}seeds[c]=State();
    let node=ids[c];let contact=contacts[c];let query=queries[c];
    if header.ready==0u || atomicLoad(&header.invalid)!=0u || node>=params.nodes || !(contact.soil_mass.w>0.0) {return;}
    var lo=0u;var hi=params.capacity;
    loop {if lo>=hi {break;}let mid=lo+(hi-lo)/2u;if history[mid].key.x<node {lo=mid+1u;}else {hi=mid;}}
    if lo>=params.capacity {return;}let old=history[lo];
    if any(old.key!=vec4<u32>(node,query.body_boundary.x,contact.boundary.x,1u)) || old.normal_mu.w!=contact.friction.x || old.mass_dt.y!=query.velocity_dt.w || !(dot(old.normal_mu.xyz,contact.normal_gap.xyz)>=0.95) {return;}
    let ratio=contact.soil_mass.w/old.mass_dt.x;
    if !(ratio>=0.5 && ratio<=2.0) {return;}
    let next=State(Six(old.state.high.k*ratio,old.state.high.w*ratio),Six(old.state.low.k*ratio,old.state.low.w*ratio));
    if !valid_six(next.high) || !valid_six(next.low) {return;}
    states[c]=next;seeds[c]=next;atomicAdd(&header.hits,1u);
    if any(next.high.k!=vec4<f32>(0.0)) || any(next.high.w!=vec4<f32>(0.0)) || any(next.low.k!=vec4<f32>(0.0)) || any(next.low.w!=vec4<f32>(0.0)) {atomicAdd(&header.nonzero,1u);}
}
@compute @workgroup_size(64)
fn save(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x;if c>=params.capacity {return;}
    if c==0u {header.ready=1u;}
    history[c]=Entry();history[c].key.x=ids[c];
    if common_status[0]!=0u || solve_status[0]!=0u || publication_status[0]!=0u {atomicOr(&header.invalid,1u);return;}
    let node=ids[c];
    if node==0xffffffffu {return;}
    if node>=params.nodes || (c>0u && ids[c-1u]>=node) {atomicOr(&header.invalid,1u);return;}
    let contact=contacts[c];let query=queries[c];let state=states[c];
    if !(contact.soil_mass.w>0.0) {return;}
    if !valid_six(state.high) || !valid_six(state.low) {atomicOr(&header.invalid,1u);return;}
    history[c]=Entry(vec4<u32>(node,query.body_boundary.x,contact.boundary.x,1u),vec4<f32>(contact.normal_gap.xyz,contact.friction.x),vec4<f32>(contact.soil_mass.w,query.velocity_dt.w,0.0,0.0),state);
}
