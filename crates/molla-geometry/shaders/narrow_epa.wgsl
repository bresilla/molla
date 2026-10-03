@group(2) @binding(1) var<storage, read_write> manifold_points: array<f32>;
@group(2) @binding(2) var<storage, read_write> manifold_counts: array<atomic<u32>>;
const MANIFOLD_POINTS: u32 = 12u;

struct PlanePair { high: vec3<f32>, low: vec3<f32> }
fn plane_round(i: u32, value: f32) -> f32 {
    atomicStore(&manifold_counts[i], bitcast<u32>(value));
    return bitcast<f32>(atomicLoad(&manifold_counts[i]));
}
fn plane_sum(i: u32, a: f32, b: f32) -> vec2<f32> {
    let sum = plane_round(i, a + b);
    let vb = plane_round(i, sum - a);
    let va = plane_round(i, sum - vb);
    let error = plane_round(i, (a - va) + (b - vb));
    return vec2<f32>(sum, error);
}
fn plane_add(i: u32, a: PlanePair, b: PlanePair) -> PlanePair {
    var result: PlanePair;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let sum = plane_sum(i, a.high[axis], b.high[axis]);
        let value = plane_sum(i, sum.x, sum.y + a.low[axis] + b.low[axis]);
        result.high[axis] = value.x;
        result.low[axis] = value.y;
    }
    return result;
}
fn plane_rotate(i: u32, q: vec4<f32>, local: vec3<f32>) -> PlanePair {
    let delta = 2.0 * cross(q.xyz, cross(q.xyz, local) + q.w * local);
    return plane_add(i, PlanePair(local, vec3<f32>(0.0)), PlanePair(delta, vec3<f32>(0.0)));
}
fn plane_center(i: u32, shape: u32) -> PlanePair {
    let local = shape_transform[shape];
    let body = shape_body[shape];
    if (body < 0) { return PlanePair(local.position.xyz, vec3<f32>(0.0)); }
    let pose = body_q[u32(body)];
    return plane_add(i, PlanePair(pose.position.xyz, plane_position_residues[u32(body)].xyz),
        plane_rotate(i, pose.rotation, local.position.xyz));
}
fn plane_depth(i: u32, shape: u32, plane: u32, local: vec3<f32>, world: vec3<f32>, n: vec3<f32>) -> f32 {
    let a = plane_center(i, shape);
    let b = plane_center(i, plane);
    var relative = plane_add(i, a, PlanePair(-b.high, -b.low));
    relative = plane_add(i, relative, plane_rotate(i, world_xf_of(shape).rotation, local));
    relative = plane_add(i, relative, PlanePair(world, vec3<f32>(0.0)));
    var total = vec2<f32>(0.0);
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let product = plane_round(i, relative.high[axis] * n[axis]);
        let tail = fma(relative.high[axis], n[axis], -product) + relative.low[axis] * n[axis];
        let sum = plane_sum(i, total.x, product);
        total = plane_sum(i, sum.x, sum.y + total.y + tail);
    }
    atomicStore(&manifold_counts[i], 0u);
    return -(total.x + total.y);
}

fn plane_rim_support(radius: f32, d: vec3<f32>) -> vec3<f32> {
    let radial_scale = max(abs(d.x), abs(d.z));
    var radial = vec2<f32>(1.0, 0.0);
    if (radial_scale > 0.0) {
        let scaled = d.xz / radial_scale;
        radial = scaled / length(scaled);
    }
    let edge = radial * radius;
    return vec3<f32>(edge.x, 0.0, edge.y);
}

fn copy_first_contact(i: u32) {
    for (var lane = 0u; lane < EPA_OUT_STRIDE; lane += 1u) {
        epa_out[i * EPA_OUT_STRIDE + lane] = manifold_points[i * MANIFOLD_POINTS * EPA_OUT_STRIDE + lane];
    }
}

fn plane_manifold(i: u32) {
    let flip = shape_geom_kind[pair_a[i]] == KIND_PLANE;
    let plane = select(pair_b[i], pair_a[i], flip);
    let shape = select(pair_a[i], pair_b[i], flip);
    let xf = world_xf_of(shape);
    let plane_xf = world_xf_of(plane);
    let n = quat_rotate(plane_xf.rotation, shape_geom_data[plane].xyz);
    let normal = select(-n, n, flip);
    let kind = shape_geom_kind[shape];
    let data = shape_geom_data[shape].xyz;
    let p = xf.position.xyz;
    let axis = quat_rotate(xf.rotation, vec3<f32>(0.0, 1.0, 0.0));
    var candidates: array<vec3<f32>, 12>;
    var depths: array<f32, 12>;
    var candidate_count = 0u;
    if (kind == KIND_SPHERE) {
        candidates[0] = p - n * data.x;
        depths[0] = plane_depth(i, shape, plane, vec3<f32>(0.0), -n * data.x, n);
        candidate_count = 1u;
    } else if (kind == KIND_CAPSULE) {
        candidates[0] = p - axis * data.x - n * data.y;
        candidates[1] = p + axis * data.x - n * data.y;
        depths[0] = plane_depth(i, shape, plane, vec3<f32>(0.0, -data.x, 0.0), -n * data.y, n);
        depths[1] = plane_depth(i, shape, plane, vec3<f32>(0.0, data.x, 0.0), -n * data.y, n);
        candidate_count = 2u;
    } else if (kind == KIND_BOX) {
        let cx = quat_rotate(xf.rotation, vec3<f32>(data.x, 0.0, 0.0));
        let cy = quat_rotate(xf.rotation, vec3<f32>(0.0, data.y, 0.0));
        let cz = quat_rotate(xf.rotation, vec3<f32>(0.0, 0.0, data.z));
        for (var j = 0u; j < 8u; j += 1u) {
            candidates[j] = p + select(-cx, cx, (j & 1u) != 0u)
                + select(-cy, cy, (j & 2u) != 0u) + select(-cz, cz, (j & 4u) != 0u);
            let local = vec3<f32>(select(-data.x, data.x, (j & 1u) != 0u),
                select(-data.y, data.y, (j & 2u) != 0u), select(-data.z, data.z, (j & 4u) != 0u));
            depths[j] = plane_depth(i, shape, plane, local, vec3<f32>(0.0), n);
        }
        candidate_count = 8u;
    } else if (kind == KIND_ELLIPSOID) {
        let d = quat_rotate(quat_conj(xf.rotation), -n);
        let scale = max(data.x, max(data.y, data.z));
        var local = vec3<f32>(0.0);
        if (scale > 0.0) {
            let weighted = (data / scale) * d;
            let norm = length(weighted);
            if (norm > 0.0) { local = data * (weighted / norm); }
        }
        candidates[0] = p + quat_rotate(xf.rotation, local);
        depths[0] = plane_depth(i, shape, plane, local, vec3<f32>(0.0), n);
        candidate_count = 1u;
    } else {
        let abs_axis = abs(axis);
        var helper = vec3<f32>(0.0, 0.0, 1.0);
        if (abs_axis.x <= abs_axis.y && abs_axis.x <= abs_axis.z) {
            helper = vec3<f32>(1.0, 0.0, 0.0);
        } else if (abs_axis.y <= abs_axis.z) { helper = vec3<f32>(0.0, 1.0, 0.0); }
        let tangent = normalize(cross(axis, helper));
        let bitangent = normalize(cross(axis, tangent));
        let top = p + axis * data.x;
        let bottom = p - axis * data.x;
        candidates[0] = top;
        depths[0] = plane_depth(i, shape, plane, vec3<f32>(0.0, data.x, 0.0), vec3<f32>(0.0), n);
        if (kind == KIND_CYLINDER) {
            candidates[1] = bottom;
            candidates[2] = top + tangent * data.y;
            candidates[3] = top - tangent * data.y;
            candidates[4] = top + bitangent * data.y;
            candidates[5] = top - bitangent * data.y;
            depths[1] = plane_depth(i, shape, plane, vec3<f32>(0.0, -data.x, 0.0), vec3<f32>(0.0), n);
            for (var j = 0u; j < 4u; j += 1u) {
                let edge = select(tangent, bitangent, j >= 2u) * select(data.y, -data.y, (j & 1u) != 0u);
                depths[2u + j] = plane_depth(i, shape, plane, vec3<f32>(0.0, data.x, 0.0), edge, n);
            }
            candidate_count = 6u;
        } else { candidate_count = 1u; }
        candidates[candidate_count] = bottom + tangent * data.y;
        candidates[candidate_count + 1u] = bottom - tangent * data.y;
        candidates[candidate_count + 2u] = bottom + bitangent * data.y;
        candidates[candidate_count + 3u] = bottom - bitangent * data.y;
        for (var j = 0u; j < 4u; j += 1u) {
            let edge = select(tangent, bitangent, j >= 2u) * select(data.y, -data.y, (j & 1u) != 0u);
            depths[candidate_count + j] = plane_depth(i, shape, plane, vec3<f32>(0.0, -data.x, 0.0), edge, n);
        }
        candidate_count += 4u;
        let d = quat_rotate(quat_conj(xf.rotation), -n);
        let rim = plane_rim_support(data.y, d);
        let disks = select(1u, 2u, kind == KIND_CYLINDER);
        for (var disk = 0u; disk < disks; disk += 1u) {
            let height = select(-data.x, data.x, kind == KIND_CYLINDER && disk == 0u);
            let lowest = p + quat_rotate(xf.rotation, rim + vec3<f32>(0.0, height, 0.0));
            let depth = plane_depth(i, shape, plane, rim + vec3<f32>(0.0, height, 0.0), vec3<f32>(0.0), n);
            if (!finite4(vec4<f32>(lowest, depth))) {
                atomicOr(&resident_status[0], FAULT_NONFINITE);
                return;
            }
            var sampled_depth = -3.4028234e38;
            if (kind == KIND_CYLINDER) { sampled_depth = depths[disk]; }
            let start = select(1u, 2u + disk * 4u, kind == KIND_CYLINDER);
            for (var j = start; j < start + 4u; j += 1u) {
                sampled_depth = max(sampled_depth, depths[j]);
            }
            if (depth > 0.0 && (sampled_depth <= 0.0 || depth - sampled_depth > 1e-6 * max(data.x, data.y))) {
                candidates[candidate_count] = lowest;
                depths[candidate_count] = depth;
                candidate_count += 1u;
            }
        }
    }
    var points: array<vec4<f32>, 12>;
    var count = 0u;
    for (var j = 0u; j < candidate_count; j += 1u) {
        let point = candidates[j];
        let depth = depths[j];
        if (!finite4(vec4<f32>(point, depth)) || !finite4(vec4<f32>(point + n * depth, 0.0))) {
            atomicOr(&resident_status[0], FAULT_NONFINITE);
            return;
        }
        if (depth > 0.0) {
            points[count] = vec4<f32>(point, depth);
            count += 1u;
        }
    }
    if (kind == KIND_BOX && count > 4u) {
        for (var j = 1u; j < count; j += 1u) {
            let value = points[j];
            var k = j;
            while (k > 0u) {
                if (points[k - 1u].w >= value.w) { break; }
                points[k] = points[k - 1u];
                k -= 1u;
            }
            points[k] = value;
        }
        count = 4u;
    }
    for (var j = 0u; j < count; j += 1u) {
        let value = points[j];
        let projected = value.xyz + n * value.w;
        let a = select(value.xyz, projected, flip);
        let b = select(projected, value.xyz, flip);
        let base = (i * MANIFOLD_POINTS + j) * EPA_OUT_STRIDE;
        for (var lane = 0u; lane < 3u; lane += 1u) {
            manifold_points[base + lane] = normal[lane];
            manifold_points[base + 4u + lane] = a[lane];
            manifold_points[base + 8u + lane] = b[lane];
        }
        manifold_points[base + 3u] = value.w;
        manifold_points[base + 12u] = bitcast<f32>(STATUS_OK);
    }
    atomicStore(&manifold_counts[i], count);
    if (count > 0u) { copy_first_contact(i); }
}

fn sphere_box_manifold(i: u32) {
    let flip = shape_geom_kind[pair_a[i]] == KIND_BOX;
    let sphere = select(pair_a[i], pair_b[i], flip);
    let box = select(pair_b[i], pair_a[i], flip);
    let sphere_xf = world_xf_of(sphere);
    let box_xf = world_xf_of(box);
    let inverse = vec4<f32>(-box_xf.rotation.xyz, box_xf.rotation.w);
    let a = plane_center(i, sphere);
    let b = plane_center(i, box);
    let relative = plane_add(i, a, PlanePair(-b.high, -b.low));
    let local = plane_add(i, plane_rotate(i, inverse, relative.high), plane_rotate(i, inverse, relative.low));
    let half = shape_geom_data[box].xyz;
    let radius = shape_geom_data[sphere].x;
    var closest = clamp(local.high, -half, half);
    var gap = plane_add(i, local, PlanePair(-closest, vec3<f32>(0.0)));
    for (var k = 0u; k < 3u; k += 1u) {
        if (abs(local.high[k]) < half[k] || (abs(local.high[k]) == half[k] && local.high[k] * local.low[k] <= 0.0)) {
            gap.high[k] = 0.0;
            gap.low[k] = 0.0;
        }
    }
    if (all(gap.high == vec3<f32>(0.0))) {
        gap.high = gap.low;
        gap.low = vec3<f32>(0.0);
    }
    let scale = max(max(abs(gap.high.x), abs(gap.high.y)), abs(gap.high.z));
    var outward = vec3<f32>(0.0);
    var depth = 0.0;
    if (scale > 0.0) {
        var scaled = gap.high / scale;
        for (var axis = 0u; axis < 3u; axis += 1u) {
            if (abs(gap.high[axis]) == scale) {
                scaled[axis] = select(-1.0, 1.0, gap.high[axis] > 0.0);
            }
        }
        let raw_magnitude = length(scaled);
        let magnitude = raw_magnitude
            + fma(-raw_magnitude, raw_magnitude, dot(scaled, scaled)) / (2.0 * raw_magnitude);
        outward = scaled / magnitude;
        let distance = plane_round(i, scale * magnitude);
        let delta = plane_sum(i, radius, -distance);
        depth = (delta.x + delta.y) - dot(outward, gap.low) - fma(scale, magnitude, -distance);
    } else {
        var axis = 0u;
        var distance = half.x - abs(local.high.x);
        for (var k = 1u; k < 3u; k += 1u) {
            let candidate = half[k] - abs(local.high[k]);
            if (candidate < distance) { axis = k; distance = candidate; }
        }
        outward[axis] = select(-1.0, 1.0, local.high[axis] >= 0.0);
        closest[axis] = outward[axis] * half[axis];
        let delta = plane_sum(i, half[axis], -abs(local.high[axis]));
        depth = radius + delta.x + delta.y - outward[axis] * local.low[axis];
    }
    atomicStore(&manifold_counts[i], 0u);
    let n = quat_rotate(box_xf.rotation, outward);
    let point_sphere = sphere_xf.position.xyz - n * radius;
    let point_box = box_xf.position.xyz + quat_rotate(box_xf.rotation, closest);
    if (!finite4(vec4<f32>(n, depth)) || !finite4(vec4<f32>(point_sphere, 0.0)) || !finite4(vec4<f32>(point_box, 0.0))) {
        atomicOr(&resident_status[0], FAULT_NONFINITE);
        return;
    }
    if (depth < 0.0) { return; }
    write_contact(i, select(-n, n, flip), depth, select(point_sphere, point_box, flip), select(point_box, point_sphere, flip), STATUS_OK);
    for (var lane = 0u; lane < EPA_OUT_STRIDE; lane += 1u) {
        manifold_points[i * MANIFOLD_POINTS * EPA_OUT_STRIDE + lane] = epa_out[i * EPA_OUT_STRIDE + lane];
    }
    atomicStore(&manifold_counts[i], 1u);
}

fn sphere_sphere_manifold(i: u32) {
    let a = plane_center(i, pair_a[i]);
    let b = plane_center(i, pair_b[i]);
    var gap = plane_add(i, b, PlanePair(-a.high, -a.low));
    if (all(gap.high == vec3<f32>(0.0))) {
        gap.high = gap.low;
        gap.low = vec3<f32>(0.0);
    }
    let scale = max(max(abs(gap.high.x), abs(gap.high.y)), abs(gap.high.z));
    var normal = vec3<f32>(select(-1.0, 1.0, pair_a[i] < pair_b[i]), 0.0, 0.0);
    var distance = 0.0;
    if (scale > 0.0) {
        let scaled = gap.high / scale;
        let magnitude = length(scaled);
        normal = scaled / magnitude;
        distance = plane_round(i, scale * magnitude);
    }
    let ra = shape_geom_data[pair_a[i]].x;
    let rb = shape_geom_data[pair_b[i]].x;
    let radii = plane_sum(i, ra, rb);
    let delta = plane_sum(i, radii.x, -distance);
    let depth = delta.x + delta.y + radii.y - dot(normal, gap.low);
    let point_a = a.high + normal * ra;
    let point_b = b.high - normal * rb;
    atomicStore(&manifold_counts[i], 0u);
    if (!finite4(vec4<f32>(normal, depth)) || !finite4(vec4<f32>(point_a, 0.0)) || !finite4(vec4<f32>(point_b, 0.0))) {
        atomicOr(&resident_status[0], FAULT_NONFINITE);
        return;
    }
    if (depth <= 0.0) { return; }
    write_contact(i, normal, depth, point_a, point_b, STATUS_OK);
    for (var lane = 0u; lane < EPA_OUT_STRIDE; lane += 1u) {
        manifold_points[i * MANIFOLD_POINTS * EPA_OUT_STRIDE + lane] = epa_out[i * EPA_OUT_STRIDE + lane];
    }
    atomicStore(&manifold_counts[i], 1u);
}

fn capsule_project(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let d = b - a;
    let square = dot(d, d);
    if (square == 0.0) { return a; }
    return a + d * clamp(dot(p - a, d) / square, 0.0, 1.0);
}

struct CapsuleClosest { a: vec3<f32>, b: vec3<f32>, interior: bool }
fn capsule_closest(a0: vec3<f32>, a1: vec3<f32>, b0: vec3<f32>, b1: vec3<f32>) -> CapsuleClosest {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let r = a0 - b0;
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);
    var s = 0.0;
    var t = 0.0;
    if (a == 0.0 && e == 0.0) { return CapsuleClosest(a0, b0, false); }
    if (a == 0.0) {
        t = clamp(f / e, 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if (e == 0.0) {
            s = clamp(-c / a, 0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let cross_axis = cross(d1, d2);
            let denominator = dot(cross_axis, cross_axis);
            if (denominator > 0.0) { s = clamp(dot(cross(d2, r), cross_axis) / denominator, 0.0, 1.0); }
            t = (b * s + f) / e;
            if (t < 0.0) {
                t = 0.0;
                s = clamp(-c / a, 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }
    return CapsuleClosest(a0 + d1 * s, b0 + d2 * t, s > 0.0 && s < 1.0 && t > 0.0 && t < 1.0);
}

fn capsule_capsule_manifold(i: u32) {
    let a = pair_a[i];
    let b = pair_b[i];
    let ca = plane_center(i, a);
    let cb = plane_center(i, b);
    let relative = plane_add(i, cb, PlanePair(-ca.high, -ca.low));
    let delta = relative.high + relative.low;
    let da = shape_geom_data[a].xy;
    let db = shape_geom_data[b].xy;
    let scale = max(max(max(da.x, da.y), max(db.x, db.y)), max(max(abs(delta.x), abs(delta.y)), abs(delta.z)));
    atomicStore(&manifold_counts[i], 0u);
    if (!(scale > 0.0) || !finite4(vec4<f32>(delta, scale))) {
        atomicOr(&resident_status[0], FAULT_NONFINITE);
        return;
    }
    let center = delta / scale;
    let axis_a = quat_rotate(world_xf_of(a).rotation, vec3<f32>(0.0, 1.0, 0.0));
    let axis_b = quat_rotate(world_xf_of(b).rotation, vec3<f32>(0.0, 1.0, 0.0));
    let a0 = -axis_a * (da.x / scale);
    let a1 = -a0;
    let b0 = center - axis_b * (db.x / scale);
    let b1 = center + axis_b * (db.x / scale);
    let ra = da.y / scale;
    let rb = db.y / scale;
    var candidates: array<CapsuleClosest, 5>;
    candidates[0] = capsule_closest(a0, a1, b0, b1);
    candidates[1] = CapsuleClosest(a0, capsule_project(a0, b0, b1), false);
    candidates[2] = CapsuleClosest(a1, capsule_project(a1, b0, b1), false);
    candidates[3] = CapsuleClosest(capsule_project(b0, a0, a1), b0, false);
    candidates[4] = CapsuleClosest(capsule_project(b1, a0, a1), b1, false);
    let cross_axis = cross(axis_a, axis_b);
    let cross_length = length(cross_axis);
    let near_parallel = cross_length <= 64.0 * 1.1920929e-7;
    var endpoint_support = false;
    for (var j = 1u; j < 5u; j += 1u) {
        if (length(candidates[j].b - candidates[j].a) < ra + rb) { endpoint_support = true; }
    }
    var normals: array<vec3<f32>, 5>;
    var midpoints: array<vec3<f32>, 5>;
    var depths: array<f32, 5>;
    var points_a: array<vec3<f32>, 5>;
    var points_b: array<vec3<f32>, 5>;
    var count = 0u;
    let tolerance = 64.0 * 1.1920929e-7 * max(max(da.x, da.y), max(db.x, db.y)) / scale;
    for (var j = 0u; j < 5u; j += 1u) {
        if (j == 0u && near_parallel && endpoint_support) { continue; }
        let gap = candidates[j].b - candidates[j].a;
        var distance = length(gap);
        var normal = vec3<f32>(0.0, select(-1.0, 1.0, a < b), 0.0);
        if (distance > 0.0) { normal = gap / distance; }
        if (candidates[j].interior && !near_parallel) {
            let perpendicular = cross_axis / cross_length;
            let signed_distance = dot(gap, perpendicular);
            normal = select(-perpendicular, perpendicular, signed_distance >= 0.0);
            distance = abs(signed_distance);
        }
        let depth = ra + rb - distance;
        if (depth <= 0.0) {
            continue;
        }
        let pa = candidates[j].a + normal * ra;
        let pb = candidates[j].b - normal * rb;
        let midpoint = (pa + pb) * 0.5;
        var duplicate = false;
        for (var k = 0u; k < count; k += 1u) {
            if (length(midpoints[k] - midpoint) <= tolerance && dot(normals[k], normal) >= 1.0 - 64.0 * 1.1920929e-7) {
                duplicate = true;
            }
        }
        if (duplicate) { continue; }
        normals[count] = normal;
        midpoints[count] = midpoint;
        depths[count] = depth;
        points_a[count] = pa;
        points_b[count] = pb;
        count += 1u;
    }
    if (count > 4u) {
        var shallowest = 0u;
        for (var j = 1u; j < count; j += 1u) {
            if (depths[j] < depths[shallowest]) { shallowest = j; }
        }
        for (var j = shallowest; j + 1u < count; j += 1u) {
            normals[j] = normals[j + 1u];
            depths[j] = depths[j + 1u];
            points_a[j] = points_a[j + 1u];
            points_b[j] = points_b[j + 1u];
        }
        count = 4u;
    }
    for (var j = 0u; j < count; j += 1u) {
        let pa = ca.high + (ca.low + points_a[j] * scale);
        let pb = ca.high + (ca.low + points_b[j] * scale);
        let depth = depths[j] * scale;
        if (!finite4(vec4<f32>(normals[j], depth)) || !finite4(vec4<f32>(pa, 0.0)) || !finite4(vec4<f32>(pb, 0.0))) {
            atomicOr(&resident_status[0], FAULT_NONFINITE);
            return;
        }
        let base = (i * MANIFOLD_POINTS + j) * EPA_OUT_STRIDE;
        for (var lane = 0u; lane < 3u; lane += 1u) {
            manifold_points[base + lane] = normals[j][lane];
            manifold_points[base + 4u + lane] = pa[lane];
            manifold_points[base + 8u + lane] = pb[lane];
        }
        manifold_points[base + 3u] = depth;
        manifold_points[base + 12u] = bitcast<f32>(STATUS_OK);
    }
    atomicStore(&manifold_counts[i], count);
    if (count > 0u) { copy_first_contact(i); }
}

@compute @workgroup_size(64)
fn resident_epa(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.pair_count) { return; }
    write_contact(i, vec3<f32>(0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_NONE);
    atomicStore(&manifold_counts[i], 0u);
    for (var lane = 0u; lane < MANIFOLD_POINTS * EPA_OUT_STRIDE; lane += 1u) {
        manifold_points[i * MANIFOLD_POINTS * EPA_OUT_STRIDE + lane] = 0.0;
    }
    if (atomicLoad(&resident_status[0]) != 0u) { return; }
    if (resident_count[0] > u.pair_count || i >= resident_count[0]) { return; }
    if (overlap_in[i] == 3u) { plane_manifold(i); return; }
    if (overlap_in[i] == 4u) { sphere_box_manifold(i); return; }
    if (overlap_in[i] == 5u) { sphere_sphere_manifold(i); return; }
    if (overlap_in[i] == 6u) { capsule_capsule_manifold(i); return; }
    if (overlap_in[i] != 1u) { return; }
    epa_pair(i);
    let base = i * EPA_OUT_STRIDE;
    if (bitcast<u32>(epa_out[base + 12u]) == STATUS_FALLBACK) {
        atomicOr(&resident_status[0], FAULT_EPA);
        return;
    }
    for (var lane = 0u; lane < 12u; lane += 4u) {
        let values = vec4<f32>(epa_out[base + lane], epa_out[base + lane + 1u],
            epa_out[base + lane + 2u], epa_out[base + lane + 3u]);
        if (!finite4(values)) {
            write_contact(i, vec3<f32>(0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_FALLBACK);
            atomicOr(&resident_status[0], FAULT_NONFINITE);
            return;
        }
    }
    if (bitcast<u32>(epa_out[base + 12u]) == STATUS_OK) {
        atomicStore(&manifold_counts[i], 1u);
        for (var lane = 0u; lane < EPA_OUT_STRIDE; lane += 1u) {
            manifold_points[i * MANIFOLD_POINTS * EPA_OUT_STRIDE + lane] = epa_out[base + lane];
        }
    }
}
