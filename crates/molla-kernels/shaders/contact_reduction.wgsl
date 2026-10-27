// GPU contact reduction — port of geometry/contact_reduction.rs.
//
// Input layout: per "group" of contacts (one shape pair), the host
// packs up to MAX_PER_GROUP contacts and the actual count into a
// dense slab. One workgroup processes one group:
//   thread 0 runs the scalar greedy algorithm (the per-group work is
//   tiny — typical inputs are 4..16 contacts capped at 4 outputs).
//
// Output: same Contact struct, up to MAX_OUT_PER_GROUP per group.
// The "deepest first" + "greedy farthest-point" policy matches the
// CPU reference verbatim (Newton-style reduction).
//
// Algorithm choice: ONE WORKGROUP PER GROUP. Per-group n is small
// (rarely > 16); workgroup-level parallelism doesn't pay. We keep
// the workgroup so future N-per-pair parallel reductions can swap
// in if needed.

// Match values must agree with the host kernel dispatcher
// (kernels/contact_reduction.rs).
const MAX_PER_GROUP: u32 = 64u;
const MAX_OUT_PER_GROUP: u32 = 4u;

struct Uniforms {
    group_count:  u32, // number of (a_shape, b_shape) groups
    max_per_pair: u32, // 1..MAX_OUT_PER_GROUP — output cap per group
    _pad0:        u32,
    _pad1:        u32,
};

// One slot per (input) contact. Laid out per-group: group `g` starts
// at `g * MAX_PER_GROUP`. Group's actual count is in `group_count_buf[g]`.
//   point_xyz | normal_xyz | depth | id_lo | id_hi | a_shape | b_shape
// → packed as 3 vec4<f32> per contact:
//   [0]: point.xyz, depth
//   [1]: normal.xyz, id_lo_as_f32
//   [2]: id_hi_as_f32, a_shape_as_f32, b_shape_as_f32, _
const CONTACT_STRIDE: u32 = 3u;

@group(0) @binding(0) var<storage, read>      in_contacts: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>      in_count:    array<u32>;

// Output: same layout as input, one MAX_OUT_PER_GROUP-sized slot per
// group. `out_count[g]` reports how many of those slots are valid.
@group(1) @binding(0) var<storage, read_write> out_contacts: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> out_count:    array<u32>;

@group(2) @binding(0) var<uniform> u: Uniforms;

// Per-workgroup scratch — kept simple (thread 0 does all the work).
var<workgroup> wg_kept: array<u32, MAX_OUT_PER_GROUP>;

fn contact_point(group: u32, local: u32) -> vec3<f32> {
    let off = (group * MAX_PER_GROUP + local) * CONTACT_STRIDE;
    return in_contacts[off + 0u].xyz;
}
fn contact_depth(group: u32, local: u32) -> f32 {
    let off = (group * MAX_PER_GROUP + local) * CONTACT_STRIDE;
    return in_contacts[off + 0u].w;
}

fn copy_contact_to_out(group: u32, src_local: u32, dst_local: u32) {
    let src = (group * MAX_PER_GROUP + src_local) * CONTACT_STRIDE;
    let dst = (group * MAX_OUT_PER_GROUP + dst_local) * CONTACT_STRIDE;
    out_contacts[dst + 0u] = in_contacts[src + 0u];
    out_contacts[dst + 1u] = in_contacts[src + 1u];
    out_contacts[dst + 2u] = in_contacts[src + 2u];
}

// Scalar reduction body for one group. Mirrors the CPU
// `reduce_one_pair` in `geometry/contact_reduction.rs`.
fn reduce_one_group(g: u32) {
    let n = min(in_count[g], MAX_PER_GROUP);
    let cap_req = min(u.max_per_pair, MAX_OUT_PER_GROUP);
    let cap = min(cap_req, n);
    if (n == 0u || cap == 0u) {
        out_count[g] = 0u;
        return;
    }

    // 1) Deepest contact wins the first slot.
    var deepest_local: u32 = 0u;
    var deepest_depth: f32 = contact_depth(g, 0u);
    for (var li: u32 = 1u; li < n; li = li + 1u) {
        let d = contact_depth(g, li);
        if (d > deepest_depth) {
            deepest_depth = d;
            deepest_local = li;
        }
    }
    var kept_count: u32 = 1u;
    wg_kept[0] = deepest_local;

    // 2) Greedy farthest-point sweep. O(n² · cap) — fine for the
    // small n we expect (≤ MAX_PER_GROUP = 64) and the tiny cap (≤ 4).
    loop {
        if (kept_count >= cap) { break; }
        var best_li: i32 = -1;
        var best_min_dist: f32 = -1.0;
        var best_depth: f32 = -1e30;
        for (var li: u32 = 0u; li < n; li = li + 1u) {
            // Skip already-kept.
            var already = false;
            for (var kk: u32 = 0u; kk < kept_count; kk = kk + 1u) {
                if (wg_kept[kk] == li) { already = true; break; }
            }
            if (already) { continue; }
            let p = contact_point(g, li);
            var min_d: f32 = 3.4028235e38;
            for (var kk: u32 = 0u; kk < kept_count; kk = kk + 1u) {
                let q = contact_point(g, wg_kept[kk]);
                let d = length(p - q);
                if (d < min_d) { min_d = d; }
            }
            let cur_depth = contact_depth(g, li);
            // Tie-break: spread first (with epsilon), then depth,
            // then lowest index. Matches CPU reduce_one_pair.
            let eps: f32 = 1e-6;
            var pick = false;
            if (best_li < 0) {
                pick = true;
            } else if (min_d > best_min_dist + eps) {
                pick = true;
            } else if (abs(min_d - best_min_dist) <= eps) {
                if (cur_depth > best_depth + eps) {
                    pick = true;
                } else if (abs(cur_depth - best_depth) <= eps) {
                    pick = (i32(li) < best_li);
                }
            }
            if (pick) {
                best_li = i32(li);
                best_min_dist = min_d;
                best_depth = cur_depth;
            }
        }
        if (best_li < 0) { break; }
        wg_kept[kept_count] = u32(best_li);
        kept_count = kept_count + 1u;
    }

    // 3) Emit kept contacts to the output slab.
    for (var k: u32 = 0u; k < kept_count; k = k + 1u) {
        copy_contact_to_out(g, wg_kept[k], k);
    }
    out_count[g] = kept_count;
}

@compute @workgroup_size(64)
fn contact_reduce(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let g = wg.x;
    if (g >= u.group_count) { return; }
    // Only thread 0 of the workgroup actually does the work — the
    // per-group reduction is scalar.
    if (lid.x != 0u) { return; }
    reduce_one_group(g);
}
