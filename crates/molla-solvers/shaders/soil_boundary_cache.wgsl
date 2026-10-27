var<workgroup> boundary_cache: array<vec3<u32>,128>;
var<private> boundary_cache_count: u32;
var<private> boundary_cache_complete: bool;

fn cache_boundary_nodes() {
    boundary_cache_count=0u;
    boundary_cache_complete=true;
    for (var b=0u;b<params.bodies;b++) {
        let span=spans[b];
        for (var k=0u;k<span.y;k++) {
            let index=span.x+k;
            let node=nodes[index];
            let mask=u32(node.momentum.w);
            if node.arm_mass.w==0.0 || mask==0u { continue; }
            if boundary_cache_count==128u {
                boundary_cache_complete=false;
                return;
            }
            boundary_cache[boundary_cache_count]=vec3<u32>(b,index,mask);
            boundary_cache_count++;
        }
    }
}
