struct CellContactPoint { a: vec4<f32>, b: vec4<f32> }
struct CellContact { normal: vec4<f32>, points: array<CellContactPoint,4> }
struct ContactPolygon { count: u32, points: array<vec3<f32>,64> }
@group(0) @binding(15) var<storage, read> contact_topology: array<u32>;
@group(0) @binding(16) var<storage, read_write> contacts: array<CellContact>;
@group(0) @binding(17) var<storage, read_write> contact_graph: array<u32>;
var<workgroup> contact_colors: u32;
var<workgroup> active_pairs: array<atomic<u32>,2>;

fn contact_index(a: u32,b: u32) -> u32 { let hi = max(a,b); return hi*(hi-1u)/2u+min(a,b); }
fn contact_bodies(pair: u32) -> vec2<u32> {
    var b = u32(floor((1.0+sqrt(1.0+8.0*f32(pair)))*0.5));
    while b*(b-1u)/2u > pair { b--; }
    while (b+1u)*b/2u <= pair { b++; }
    return vec2<u32>(pair-b*(b-1u)/2u,b);
}
fn contact_pair_list() -> u32 {
    let n = params.counts.x;
    return 3u+4u*n+n*((2u*n+31u)/32u);
}
fn cell_vertex(i: u32,v: u32,origin: vec3<f32>) -> vec3<f32> {
    let m = motions[i]; let f = fibers[i];
    return m.center.xyz-origin+rot(m.rotation,vertices[f.vertices.x+v].xyz-f.com_mass.xyz);
}
fn cell_face_normal(i: u32,face: u32) -> vec3<f32> {
    return rot(motions[i].rotation,bitcast<vec3<f32>>(vec3<u32>(contact_topology[face+1u],contact_topology[face+2u],contact_topology[face+3u])));
}
fn cell_interval(i: u32,n: vec3<f32>,origin: vec3<f32>) -> vec2<f32> {
    var bounds = vec2<f32>(3.402823e38,-3.402823e38);
    for (var v = 0u; v < fibers[i].vertices.y; v++) {
        let d = dot(cell_vertex(i,v,origin),n); bounds = vec2<f32>(min(bounds.x,d),max(bounds.y,d));
    }
    return bounds;
}
fn contact_push(polygon: ptr<function,ContactPolygon>,p: vec3<f32>) {
    if (*polygon).count >= 64u { atomicOr(&status[0],8u); return; }
    (*polygon).points[(*polygon).count] = p; (*polygon).count++;
}
fn contact_clip(polygon: ContactPolygon,origin: vec3<f32>,n: vec3<f32>) -> ContactPolygon {
    var clipped: ContactPolygon;
    for (var i = 0u; i < polygon.count; i++) {
        let a = polygon.points[i]; let b = polygon.points[(i+1u)%polygon.count];
        let da = dot(a-origin,n); let db = dot(b-origin,n);
        if da <= 0.0 { contact_push(&clipped,a); }
        if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) { contact_push(&clipped,mix(a,b,da/(da-db))); }
    }
    return clipped;
}
fn cell_support_segment(i: u32,n: vec3<f32>,origin: vec3<f32>,tolerance: f32) -> mat2x3<f32> {
    let extreme = cell_interval(i,n,origin).y;
    var segment: mat2x3<f32>; var length = -1.0;
    for (var j = 0u; j < fibers[i].vertices.y; j++) {
        let a = cell_vertex(i,j,origin); if extreme-dot(a,n) > tolerance { continue; }
        for (var k = j; k < fibers[i].vertices.y; k++) {
            let b = cell_vertex(i,k,origin); if extreme-dot(b,n) > tolerance { continue; }
            let distance = dot(a-b,a-b);
            if distance > length { length = distance; segment = mat2x3<f32>(a,b); }
        }
    }
    return segment;
}
fn contact_segment_witnesses(a: mat2x3<f32>,b: mat2x3<f32>) -> mat2x3<f32> {
    let u = a[1]-a[0]; let v = b[1]-b[0]; let r = a[0]-b[0];
    let aa = dot(u,u); let bb = dot(v,v); let ab = dot(u,v); let ar = dot(u,r); let br = dot(v,r);
    let determinant = aa*bb-ab*ab; var s = 0.0; var t = 0.0;
    if determinant > 1e-12*aa*bb { s = clamp((ab*br-ar*bb)/determinant,0.0,1.0); }
    if bb > 0.0 { t = (ab*s+br)/bb; }
    if t < 0.0 { t = 0.0; if aa > 0.0 { s = clamp(-ar/aa,0.0,1.0); } }
    else if t > 1.0 { t = 1.0; if aa > 0.0 { s = clamp((ab-ar)/aa,0.0,1.0); } }
    else if bb == 0.0 && aa > 0.0 { s = clamp(-ar/aa,0.0,1.0); }
    return mat2x3<f32>(a[0]+u*s,b[0]+v*t);
}
fn contact_witness(ai: u32,bi: u32,p: vec3<f32>,q: vec3<f32>,n: vec3<f32>,origin: vec3<f32>) -> CellContactPoint {
    let a = motions[ai]; let b = motions[bi];
    let ra = p+origin-a.center.xyz; let rb = q+origin-b.center.xyz;
    let va = a.velocity.xyz+cross(a.omega.xyz,ra); let vb = b.velocity.xyz+cross(b.omega.xyz,rb);
    return CellContactPoint(vec4<f32>(rot(conj(a.rotation),ra)+fibers[ai].com_mass.xyz,0.0),
        vec4<f32>(rot(conj(b.rotation),rb)+fibers[bi].com_mass.xyz,dot(vb-va,n)));
}
fn cell_manifold_gpu(ai: u32,bi: u32) -> CellContact {
    var result: CellContact;
    let fa = fibers[ai]; let fb = fibers[bi];
    let origin = motions[ai].center.xyz-rot(motions[ai].rotation,fa.com_mass.xyz);
    let margin = params.coupled.z;
    for (var axis = 0u; axis < 3u; axis++) {
        var n = vec3<f32>(0.0); n[axis] = 1.0;
        let a = cell_interval(ai,n,origin); let b = cell_interval(bi,n,origin);
        if b.x-a.y > margin || a.x-b.y > margin { return result; }
    }
    var scale = 0.0;
    for (var side = 0u; side < 2u; side++) {
        let i = select(ai,bi,side == 1u);
        for (var v = 0u; v < fibers[i].vertices.y; v++) { scale = max(scale,length(cell_vertex(i,v,origin))); }
    }
    let tolerance = 1e-6*scale;
    var best = -3.402823e38; var normal = vec3<f32>(0.0); var reference = 0xffffffffu; var reference_side = 0u;
    for (var side = 0u; side < 2u; side++) {
        let i = select(ai,bi,side == 1u); let other = select(bi,ai,side == 1u);
        var face = fibers[i].surface.x;
        for (var j = 0u; j < fibers[i].surface.y; j++) {
            let n = cell_face_normal(i,face); let p = cell_vertex(i,contact_topology[face+4u],origin);
            let separation = cell_interval(other,n,origin).x-dot(p,n);
            if separation > margin { return result; }
            if separation > best+tolerance {
                best = separation; normal = n*select(1.0,-1.0,side == 1u); reference = face; reference_side = side;
            }
            face += 4u+contact_topology[face];
        }
    }
    for (var ea = 0u; ea < fa.surface.w; ea++) {
        let va = fa.surface.z+2u*ea;
        let u = normalize(cell_vertex(ai,contact_topology[va+1u],origin)-cell_vertex(ai,contact_topology[va],origin));
        for (var eb = 0u; eb < fb.surface.w; eb++) {
            let vb = fb.surface.z+2u*eb;
            let v = normalize(cell_vertex(bi,contact_topology[vb+1u],origin)-cell_vertex(bi,contact_topology[vb],origin));
            let axis = cross(u,v); if dot(axis,axis) < 1e-12 { continue; }
            var n = normalize(axis); let a = cell_interval(ai,n,origin); let b = cell_interval(bi,n,origin);
            var separation = b.x-a.y; if a.x-b.y > separation { separation = a.x-b.y; n = -n; }
            if separation > margin { return result; }
            if separation > best+tolerance { best = separation; normal = n; reference = 0xffffffffu; }
        }
    }
    if !finite4(vec4<f32>(normal,best)) { atomicOr(&status[0],8u); return result; }
    result.normal = vec4<f32>(normal,0.0);
    if reference == 0xffffffffu {
        let points = contact_segment_witnesses(cell_support_segment(ai,normal,origin,tolerance),cell_support_segment(bi,-normal,origin,tolerance));
        result.points[0] = contact_witness(ai,bi,points[0],points[1],normal,origin); result.normal.w = 1.0; return result;
    }
    let ri = select(ai,bi,reference_side == 1u); let ii = select(bi,ai,reference_side == 1u);
    let n = normal*select(1.0,-1.0,reference_side == 1u);
    var incident = fibers[ii].surface.x; var least = 3.402823e38; var face = incident;
    for (var j = 0u; j < fibers[ii].surface.y; j++) {
        let d = dot(cell_face_normal(ii,face),n);
        if d < least { least = d; incident = face; }
        face += 4u+contact_topology[face];
    }
    var polygon: ContactPolygon;
    for (var j = 0u; j < contact_topology[incident]; j++) { contact_push(&polygon,cell_vertex(ii,contact_topology[incident+4u+j],origin)); }
    let count = contact_topology[reference]; let root = cell_vertex(ri,contact_topology[reference+4u],origin);
    for (var j = 0u; j < count; j++) {
        let a = cell_vertex(ri,contact_topology[reference+4u+j],origin);
        let b = cell_vertex(ri,contact_topology[reference+4u+(j+1u)%count],origin);
        polygon = contact_clip(polygon,a,cross(b-a,n));
    }
    var candidates: ContactPolygon; var midpoints: array<vec3<f32>,64>; var depths: array<f32,64>;
    for (var j = 0u; j < polygon.count; j++) {
        let p = polygon.points[j]; let distance = dot(p-root,n);
        if distance <= margin {
            let k = candidates.count; contact_push(&candidates,p); depths[k] = -distance; midpoints[k] = p-n*(0.5*distance);
        }
    }
    var chosen: array<u32,4>;
    for (var k = 0u; k < min(candidates.count,4u); k++) {
        var pick = 0xffffffffu; var spread = -1.0; var depth = -3.402823e38;
        for (var j = 0u; j < candidates.count; j++) {
            var used = false; var distance = 3.402823e38;
            for (var l = 0u; l < k; l++) {
                used = used || chosen[l] == j; distance = min(distance,length(midpoints[j]-midpoints[chosen[l]]));
            }
            if used { continue; }
            if pick == 0xffffffffu || (k == 0u && depths[j] > depth+tolerance)
                || (k > 0u && (distance > spread+tolerance || (abs(distance-spread) <= tolerance && depths[j] > depth+tolerance))) {
                pick = j; spread = distance; depth = depths[j];
            }
        }
        chosen[k] = pick; let p = candidates.points[pick]; let q = p+depths[pick]*n;
        let a = select(q,p,reference_side == 1u); let b = select(p,q,reference_side == 1u);
        result.points[k] = contact_witness(ai,bi,a,b,normal,origin); result.normal.w += 1.0;
    }
    return result;
}
fn detect_contact_pair(pair: u32) {
    for (var i = pair; i < contact_pair_list(); i += params.counts.w) { contact_graph[i] = 0u; }
    let bodies = contact_bodies(pair); let ai = bodies.x; let bi = bodies.y; let fa = fibers[ai]; let fb = fibers[bi];
    var empty: CellContact; contacts[pair] = empty;
    if fa.com_mass.w == 0.0 && fb.com_mass.w == 0.0 { return; }
    let reach = 1.01*(fa.inertia0.w+fb.inertia0.w)+params.coupled.z;
    let gap = motions[ai].center.xyz-motions[bi].center.xyz;
    if dot(gap,gap) > reach*reach { return; }
    for (var j = 0u; j < fa.vertices.w; j++) {
        let edge = colors[fa.vertices.z+j]; let b = bonds[edge/2u];
        if bond_states[edge/2u].damage.x >= 1.0 { continue; }
        if b.fibers.x == bi || b.fibers.y == bi { return; }
        if params.numerics.y >= 2.0 {
            let other = select(b.fibers.x,b.fibers.y,b.fibers.x == ai);
            let neighbor = fibers[other];
            for (var k = 0u; k < neighbor.vertices.w; k++) {
                let link = colors[neighbor.vertices.z+k]/2u;
                if bond_states[link].damage.x < 1.0 && (bonds[link].fibers.x == bi || bonds[link].fibers.y == bi) { return; }
            }
        }
    }
    contacts[pair] = cell_manifold_gpu(ai,bi);
}
@compute @workgroup_size(64)
fn detect_contacts(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    if local.x < 2u { atomicStore(&active_pairs[local.x],0u); }
    workgroupBarrier();
    if id.x < params.counts.w && atomicLoad(&status[0]) == 0u {
        detect_contact_pair(id.x);
        if contacts[id.x].normal.w > 0.0 { atomicOr(&active_pairs[local.x/32u],1u<<(local.x%32u)); }
    }
    workgroupBarrier();
    let word = id.x/64u*2u+local.x;
    if local.x < 2u && word < (params.counts.w+31u)/32u {
        contact_graph[contact_pair_list()+params.counts.w+word] = atomicLoad(&active_pairs[local.x]);
    }
}
var<workgroup> active_counts: array<u32,256>;
// Ascending indices of non-zero active-pair words, written after the active bitmask.
fn active_words_base() -> u32 { return contact_pair_list()+params.counts.w; }
fn compact_words_base() -> u32 { return active_words_base()+(params.counts.w+31u)/32u; }
@compute @workgroup_size(256)
fn compact_contacts(@builtin(local_invocation_id) id: vec3<u32>) {
    let words = (params.counts.w+31u)/32u; let chunk = (words+255u)/256u;
    let first = min(id.x*chunk,words); let last = min(first+chunk,words);
    var count = 0u;
    for (var word = first; word < last; word++) { count += select(0u,1u,contact_graph[active_words_base()+word] != 0u); }
    active_counts[id.x] = count;
    workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var value = active_counts[id.x];
        if id.x >= stride { value += active_counts[id.x-stride]; }
        workgroupBarrier();
        active_counts[id.x] = value;
        workgroupBarrier();
    }
    var slot = active_counts[id.x]-count;
    for (var word = first; word < last; word++) {
        if contact_graph[active_words_base()+word] != 0u { contact_graph[compact_words_base()+1u+slot] = word; slot++; }
    }
    if id.x == 255u { contact_graph[compact_words_base()] = active_counts[255]; }
}
@compute @workgroup_size(1)
fn color_contacts() {
    if atomicLoad(&status[0]) != 0u { return; }
    let n = params.counts.x; let words = (2u*n+31u)/32u;
    let counts = 3u+2u*n; let masks = 3u+4u*n; let list = contact_pair_list();
    var colors = 0u;
    for (var entry = 0u; entry < contact_graph[compact_words_base()]; entry++) {
      let block = contact_graph[compact_words_base()+1u+entry];
      var occupied = contact_graph[active_words_base()+block];
      while occupied != 0u {
        let pair = block*32u+firstTrailingBit(occupied); occupied &= occupied-1u;
        let points = u32(contacts[pair].normal.w);
        let bodies = contact_bodies(pair); var color = 2u*n;
        for (var word = 0u; word < words; word++) {
            let a = masks+bodies.x*words+word; let b = masks+bodies.y*words+word;
            let available = ~(contact_graph[a]|contact_graph[b]);
            if available != 0u {
                let bit = firstTrailingBit(available); color = word*32u+bit;
                contact_graph[a] |= 1u<<bit; contact_graph[b] |= 1u<<bit; break;
            }
        }
        if color >= 2u*n { atomicOr(&status[0],8u); return; }
        colors = max(colors,color+1u); contact_graph[counts+color]++;
        contacts[pair].normal.w = f32(points|(color<<3u));
      }
    }
    var total = 0u;
    for (var color = 0u; color < colors; color++) {
        let count = contact_graph[counts+color]; contact_graph[2u+color] = total;
        contact_graph[counts+color] = total; total += count;
    }
    contact_graph[2u+colors] = total; contact_graph[0] = colors; contact_graph[1] = total;
    for (var entry = 0u; entry < contact_graph[compact_words_base()]; entry++) {
      let block = contact_graph[compact_words_base()+1u+entry];
      var occupied = contact_graph[active_words_base()+block];
      while occupied != 0u {
        let pair = block*32u+firstTrailingBit(occupied); occupied &= occupied-1u;
        let packed = u32(contacts[pair].normal.w);
        let color = packed>>3u; let index = contact_graph[counts+color];
        contact_graph[list+index] = pair; contact_graph[counts+color]++;
      }
    }
}
fn contact_row(n: vec3<f32>,p: CellContactPoint,a: Motion,b: Motion,fa: Fiber,fb: Fiber) -> Row {
    let ra = rot(a.rotation,p.a.xyz-fa.com_mass.xyz); let rb = rot(b.rotation,p.b.xyz-fb.com_mass.xyz);
    return Row(dot((b.center.xyz-a.center.xyz)+(rb-ra),n),0.0,-n,n,-cross(ra,n),cross(rb,n));
}
fn project_contact_pair(ai: u32,bi: u32) {
    let pair = contact_index(ai,bi); var c = contacts[pair]; if c.normal.w == 0.0 { return; }
    let fa = fibers[ai]; let fb = fibers[bi]; var a = motions[ai]; var b = motions[bi];
    for (var k = 0u; k < (u32(c.normal.w)&7u); k++) {
        let r = contact_row(c.normal.xyz,c.points[k],a,b,fa,fb); let w = weight(r,a,b,fa,fb); if w == 0.0 { continue; }
        let lambda = max(0.0,c.points[k].a.w-r.value/w); var dl = lambda-c.points[k].a.w;
        if !finite4(vec4<f32>(lambda,w,0.0,0.0)) { atomicOr(&status[0],8u); return; }
        let angle = max(length(inverse_apply(fa,a.rotation,r.aa)),length(inverse_apply(fb,b.rotation,r.ab)))*abs(dl);
        if angle > 0.2 { dl *= 0.2/angle; }
        c.points[k].a.w += dl; a = correction(a,fa,r.la,r.aa,dl); b = correction(b,fb,r.lb,r.ab,dl);
    }
    motions[ai] = a; motions[bi] = b; contacts[pair] = c;
}
@compute @workgroup_size(64)
fn project_contacts(@builtin(local_invocation_id) id: vec3<u32>) {
    if id.x == 0u { contact_colors = contact_graph[0]; }
    let count = workgroupUniformLoad(&contact_colors); let list = contact_pair_list();
    for (var color = 0u; color < count; color++) {
        for (var index = contact_graph[2u+color]+id.x; index < contact_graph[3u+color]; index += 64u) {
            let pair = contact_bodies(contact_graph[list+index]);
            if atomicLoad(&status[0]) == 0u { project_contact_pair(pair.x,pair.y); }
        }
        storageBarrier();
    }
}
fn contact_velocity_pair(ai: u32,bi: u32) {
    let c = contacts[contact_index(ai,bi)]; if c.normal.w == 0.0 { return; }
    let fa = fibers[ai]; let fb = fibers[bi]; var a = motions[ai]; var b = motions[bi];
    var normal_impulses: array<f32,4>;
    for (var sweep = 0u; sweep < 12u; sweep++) {
        for (var k = 0u; k < (u32(c.normal.w)&7u); k++) {
            let p = c.points[k]; if p.a.w == 0.0 { continue; }
            let r = contact_row(c.normal.xyz,p,a,b,fa,fb); let w = weight(r,a,b,fa,fb); if w == 0.0 { continue; }
            let rate = dot(r.la,a.velocity.xyz)+dot(r.lb,b.velocity.xyz)+dot(r.aa,a.omega.xyz)+dot(r.ab,b.omega.xyz);
            let desired = -params.coupled.w*min(0.0,p.b.w); let impulse = (desired-rate)/w;
            normal_impulses[k] += impulse;
            a.velocity += vec4<f32>(r.la*(fa.com_mass.w*impulse),0.0); b.velocity += vec4<f32>(r.lb*(fb.com_mass.w*impulse),0.0);
            a.omega += vec4<f32>(inverse_apply(fa,a.rotation,r.aa)*impulse,0.0); b.omega += vec4<f32>(inverse_apply(fb,b.rotation,r.ab)*impulse,0.0);
        }
    }
    for (var k = 0u; k < (u32(c.normal.w)&7u); k++) {
        let p = c.points[k]; if p.a.w == 0.0 { continue; }
        let ra = rot(a.rotation,p.a.xyz-fa.com_mass.xyz); let rb = rot(b.rotation,p.b.xyz-fb.com_mass.xyz);
        let relative = b.velocity.xyz+cross(b.omega.xyz,rb)-a.velocity.xyz-cross(a.omega.xyz,ra);
        let tangent = relative-c.normal.xyz*dot(relative,c.normal.xyz); let speed = length(tangent);
        if speed > 0.0 {
            let t = contact_row(tangent/speed,p,a,b,fa,fb); let wt = weight(t,a,b,fa,fb);
            let friction_impulse = -min(speed/wt,params.step.w*max(0.0,p.a.w/params.step.x+normal_impulses[k]));
            a.velocity += vec4<f32>(t.la*(fa.com_mass.w*friction_impulse),0.0); b.velocity += vec4<f32>(t.lb*(fb.com_mass.w*friction_impulse),0.0);
            a.omega += vec4<f32>(inverse_apply(fa,a.rotation,t.aa)*friction_impulse,0.0); b.omega += vec4<f32>(inverse_apply(fb,b.rotation,t.ab)*friction_impulse,0.0);
        }
    }
    motions[ai] = a; motions[bi] = b;
}
@compute @workgroup_size(64)
fn contact_velocities(@builtin(local_invocation_id) id: vec3<u32>) {
    if id.x == 0u { contact_colors = contact_graph[0]; }
    let count = workgroupUniformLoad(&contact_colors); let list = contact_pair_list();
    for (var color = 0u; color < count; color++) {
        for (var index = contact_graph[2u+color]+id.x; index < contact_graph[3u+color]; index += 64u) {
            let pair = contact_bodies(contact_graph[list+index]);
            if atomicLoad(&status[0]) == 0u { contact_velocity_pair(pair.x,pair.y); }
        }
        storageBarrier();
    }
}
