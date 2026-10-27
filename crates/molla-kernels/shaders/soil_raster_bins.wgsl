struct Params {
    dims: vec4<u32>,
    level: vec4<u32>,
    terrain: vec4<f32>,
    velocity: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Params;
@group(1) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> velocities: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> heights: array<f32>;
@group(1) @binding(4) var<storage, read_write> sinkage: array<f32>;
@group(1) @binding(5) var<storage, read_write> surface_velocity: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read_write> bins: array<atomic<u32>>;
@group(1) @binding(7) var<storage, read_write> members: array<u32>;

var<workgroup> scan_values: array<u32, 64>;

fn group_index(wg: vec3<u32>) -> u32 {
    return wg.x + wg.y * u.dims.w;
}

@compute @workgroup_size(64)
fn histogram(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let p = group_index(wg) * 64u + lane;
    if p < u.dims.x {
        atomicAdd(&bins[cell_index(positions[p]) * 3u], 1u);
    }
}

@compute @workgroup_size(64)
fn scan_blocks(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let block = group_index(wg);
    let item = block * 64u + lane;
    var value = 0u;
    if item < u.level.y {
        value = atomicLoad(&bins[(u.level.x + item) * 3u]);
    }
    scan_values[lane] = value;
    workgroupBarrier();
    for (var stride = 1u; stride < 64u; stride *= 2u) {
        var addend = 0u;
        if lane >= stride {
            addend = scan_values[lane - stride];
        }
        workgroupBarrier();
        scan_values[lane] += addend;
        workgroupBarrier();
    }
    if item < u.level.y {
        atomicStore(&bins[(u.level.x + item) * 3u + 1u], scan_values[lane] - value);
    }
    if lane == 63u && u.level.w != 0u && block < (u.level.y + 63u) / 64u {
        atomicStore(&bins[(u.level.z + block) * 3u], scan_values[63u]);
    }
}

@compute @workgroup_size(64)
fn scan_fixup(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let item = group_index(wg) * 64u + lane;
    if item < u.level.y {
        let parent = atomicLoad(&bins[(u.level.z + item / 64u) * 3u + 1u]);
        atomicAdd(&bins[(u.level.x + item) * 3u + 1u], parent);
    }
}

@compute @workgroup_size(64)
fn scatter(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let p = group_index(wg) * 64u + lane;
    if p < u.dims.x {
        let cell = cell_index(positions[p]) * 3u;
        let start = atomicLoad(&bins[cell + 1u]);
        let offset = atomicAdd(&bins[cell + 2u], 1u);
        members[start + offset] = p;
    }
}
