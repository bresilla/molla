// GPU marching cubes. Mirrors the CPU `marching_cubes` in
// `crates/molla-geometry/src/marching_cubes.rs`: the Lorensen-Cline
// 256-case lookup, one thread per grid cell. The EDGE/TRI tables are
// uploaded from the same Rust constants, so only EDGE_VERTS is inlined.
//
// Per cell: up to 5 triangles are written into a dense 45-float slot
// and the triangle count into out_count; the host compacts the cells in
// order to reproduce the CPU mesh exactly.

struct Uniforms {
    dims: vec4<u32>,     // (nx, ny, nz, cell_count)
    origin: vec4<f32>,   // xyz origin, w = cell_size
    iso: vec4<f32>,      // x = iso
};

@group(0) @binding(0) var<storage, read> sdf: array<f32>;
@group(0) @binding(1) var<storage, read> edge_table: array<u32>;
@group(0) @binding(2) var<storage, read> tri_table: array<i32>;

@group(1) @binding(0) var<storage, read_write> out_verts: array<f32>;
@group(1) @binding(1) var<storage, read_write> out_count: array<u32>;

@group(2) @binding(0) var<uniform> u: Uniforms;

// Edge → corner-pair (standard MC convention).
const EV0 = array<u32, 12>(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u);
const EV1 = array<u32, 12>(1u, 2u, 3u, 0u, 5u, 6u, 7u, 4u, 4u, 5u, 6u, 7u);

fn sdf_at(x: u32, y: u32, z: u32) -> f32 {
    let nx = u.dims.x;
    let ny = u.dims.y;
    return sdf[x + y * nx + z * nx * ny];
}

fn voxel_pos(x: u32, y: u32, z: u32) -> vec3<f32> {
    let cell = u.origin.w;
    return u.origin.xyz + vec3<f32>(f32(x), f32(y), f32(z)) * cell;
}

fn interp(iso: f32, a: vec3<f32>, b: vec3<f32>, va: f32, vb: f32) -> vec3<f32> {
    if (abs(iso - va) < 1.0e-9) { return a; }
    if (abs(iso - vb) < 1.0e-9) { return b; }
    if (abs(vb - va) < 1.0e-9) { return a; }
    let t = (iso - va) / (vb - va);
    return a + (b - a) * t;
}

@compute @workgroup_size(64)
fn march(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    let cell_count = u.dims.w;
    if (cell >= cell_count) { return; }

    let cnx = u.dims.x - 1u;
    let cny = u.dims.y - 1u;
    let x = cell % cnx;
    let y = (cell / cnx) % cny;
    let z = cell / (cnx * cny);

    out_count[cell] = 0u;

    var cxs = array<u32, 8>(x, x + 1u, x + 1u, x, x, x + 1u, x + 1u, x);
    var cys = array<u32, 8>(y, y, y + 1u, y + 1u, y, y, y + 1u, y + 1u);
    var czs = array<u32, 8>(z, z, z, z, z + 1u, z + 1u, z + 1u, z + 1u);

    let iso = u.iso.x;
    var vals: array<f32, 8>;
    var pos: array<vec3<f32>, 8>;
    var cube_index: u32 = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let v = sdf_at(cxs[i], cys[i], czs[i]);
        vals[i] = v;
        pos[i] = voxel_pos(cxs[i], cys[i], czs[i]);
        if (v < iso) { cube_index = cube_index | (1u << i); }
    }

    let edges = edge_table[cube_index];
    if (edges == 0u) { return; }

    var vlist: array<vec3<f32>, 12>;
    for (var e = 0u; e < 12u; e = e + 1u) {
        if ((edges & (1u << e)) != 0u) {
            let i = EV0[e];
            let j = EV1[e];
            vlist[e] = interp(iso, pos[i], pos[j], vals[i], vals[j]);
        }
    }

    var count = 0u;
    let row = cube_index * 16u;
    for (var k = 0u; k < 16u; k = k + 3u) {
        let e0 = tri_table[row + k];
        if (e0 < 0) { break; }
        let e1 = tri_table[row + k + 1u];
        let e2 = tri_table[row + k + 2u];
        let base = cell * 45u + count * 9u;
        let p0 = vlist[u32(e0)];
        let p1 = vlist[u32(e1)];
        let p2 = vlist[u32(e2)];
        out_verts[base + 0u] = p0.x;
        out_verts[base + 1u] = p0.y;
        out_verts[base + 2u] = p0.z;
        out_verts[base + 3u] = p1.x;
        out_verts[base + 4u] = p1.y;
        out_verts[base + 5u] = p1.z;
        out_verts[base + 6u] = p2.x;
        out_verts[base + 7u] = p2.y;
        out_verts[base + 8u] = p2.z;
        count = count + 1u;
    }
    out_count[cell] = count;
}
