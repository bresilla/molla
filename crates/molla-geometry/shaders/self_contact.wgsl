// GPU vertex-triangle self-contact candidate generation. Mirrors the
// CPU `detect_vertex_triangle_self_contacts` in
// `crates/molla-geometry/src/self_contact.rs`: every vertex is tested
// against every non-adjacent triangle, and a candidate is emitted when
// the vertex lies inside the thickness shell.
//
// One thread per (vertex, triangle) pair. The dense output array holds
// one STRIDE-float slot per pair; slot[11] < 0 marks "no candidate"
// (adjacent face or outside the shell). The host compacts and sorts the
// valid slots to match the CPU list exactly.

struct Uniforms {
    counts: vec4<u32>,   // (vertex_count, tri_count, pair_count, _)
    params: vec4<f32>,   // (thickness, _, _, _)
};

@group(0) @binding(0) var<storage, read> verts: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tris:  array<u32>;

@group(1) @binding(0) var<storage, read_write> out: array<f32>;

@group(2) @binding(0) var<uniform> u: Uniforms;

const STRIDE: u32 = 12u;

// Closest point on triangle (a, b, c) to p — Ericson §5.1.5.
fn closest_pt_tri(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return a; }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return b; }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return c; }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return a + ab * v + ac * w;
}

@compute @workgroup_size(64)
fn detect_self_contact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let tri_count = u.counts.y;
    let pair_count = u.counts.z;
    if (i >= pair_count) { return; }

    let base = i * STRIDE;
    out[base + 11u] = -1.0; // sentinel: no candidate

    let vi = i / tri_count;
    let ti = i % tri_count;
    let t0 = tris[3u * ti + 0u];
    let t1 = tris[3u * ti + 1u];
    let t2 = tris[3u * ti + 2u];
    // Skip incident faces (the vertex belongs to this triangle).
    if (t0 == vi || t1 == vi || t2 == vi) { return; }

    let p = verts[vi].xyz;
    let a = verts[t0].xyz;
    let b = verts[t1].xyz;
    let c = verts[t2].xyz;
    let q = closest_pt_tri(p, a, b, c);
    let dpq = p - q;
    let d2 = dot(dpq, dpq);
    let thickness = u.params.x;
    if (d2 > thickness * thickness) { return; }

    let distance = sqrt(d2);
    let face_n = cross(b - a, c - a);
    var normal = vec3<f32>(0.0, 0.0, 1.0);
    if (distance > 1.0e-12) {
        normal = dpq / distance;
    } else {
        let len = length(face_n);
        if (len > 1.0e-12) { normal = face_n / len; }
    }

    out[base + 0u] = f32(vi);
    out[base + 1u] = f32(t0);
    out[base + 2u] = f32(t1);
    out[base + 3u] = f32(t2);
    out[base + 4u] = q.x;
    out[base + 5u] = q.y;
    out[base + 6u] = q.z;
    out[base + 7u] = distance - thickness;
    out[base + 8u] = normal.x;
    out[base + 9u] = normal.y;
    out[base + 10u] = normal.z;
    out[base + 11u] = distance;
}
