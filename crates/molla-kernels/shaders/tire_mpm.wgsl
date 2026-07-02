struct ContactCell {
    center_radius: vec4<f32>,
    force_pressure: vec4<f32>,
};

struct Uniforms {
    dims: vec4<u32>,
    // x: particle_mass, y: dt, z: plastic_compaction_gain, w: max_compaction
    params: vec4<f32>,
    // x: damping
    params2: vec4<f32>,
};

struct RasterUniforms {
    dims: vec4<u32>,
    terrain0: vec4<f32>,
    terrain_velocity: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> pos_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> vel_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> compact_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> contacts: array<ContactCell>;
@group(0) @binding(4) var<storage, read_write> pos_out: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> vel_out: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> compact_out: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> u: Uniforms;

@group(2) @binding(0) var<storage, read_write> pos_rw: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> vel_rw: array<vec4<f32>>;
@group(2) @binding(2) var<storage, read_write> compact_rw: array<vec4<f32>>;
@group(2) @binding(3) var<storage, read> contacts_in_place: array<ContactCell>;

@group(3) @binding(0) var<storage, read> raster_pos: array<vec4<f32>>;
@group(3) @binding(1) var<storage, read> raster_compact: array<vec4<f32>>;
@group(3) @binding(2) var<storage, read_write> raster_heights: array<f32>;
@group(3) @binding(3) var<storage, read_write> raster_sinkage: array<f32>;
@group(3) @binding(4) var<storage, read> raster_vel: array<vec4<f32>>;
@group(3) @binding(5) var<storage, read_write> raster_surface_velocity: array<vec4<f32>>;
@group(4) @binding(0) var<uniform> ru: RasterUniforms;

@compute @workgroup_size(64)
fn apply_feedback(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.dims.x) {
        return;
    }
    let mass = max(u.params.x, 1.0e-9);
    let dt = max(u.params.y, 0.0);
    let gain = max(u.params.z, 0.0);
    let max_compaction = max(u.params.w, 0.0);
    let damping = max(u.params2.x, 0.0);

    var p = pos_in[i];
    var v = vel_in[i];
    var compact = compact_in[i];
    var dv = vec3<f32>(0.0, 0.0, 0.0);
    var accum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var damping_weight = 0.0;

    var c = 0u;
    loop {
        if (c >= u.dims.y) {
            break;
        }
        let contact = contacts[c];
        let influence = max(contact.center_radius.w, 1.0e-6);
        let dx = p.x - contact.center_radius.x;
        let dz = p.z - contact.center_radius.z;
        let horizontal = sqrt(dx * dx + dz * dz);
        if (horizontal <= influence) {
            let below_surface = contact.center_radius.y - p.y;
            if (below_surface >= -influence * 0.5 && below_surface <= influence * 2.0) {
                let radial_w = clamp(1.0 - horizontal / influence, 0.0, 1.0);
                let depth_w = clamp(1.0 - max(below_surface, 0.0) / (influence * 2.0), 0.10, 1.0);
                let w = radial_w * radial_w * depth_w;
                let shear = length(vec2<f32>(contact.force_pressure.x, contact.force_pressure.z));
                let pressure = max(contact.force_pressure.w, max(-contact.force_pressure.y, 0.0));
                dv += contact.force_pressure.xyz * (w * dt / mass);
                accum.x += pressure * w * gain * dt;
                accum.y += max(-contact.force_pressure.y, 0.0) * w * dt;
                accum.z += shear * w * dt;
                accum.w += w;
                damping_weight = max(damping_weight, w);
            }
        }
        c += 1u;
    }

    let damp = clamp(damping * damping_weight * dt, 0.0, 0.95);
    v.x = (v.x + dv.x) * (1.0 - damp);
    v.y = (v.y + dv.y) * (1.0 - damp);
    v.z = (v.z + dv.z) * (1.0 - damp);
    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;
    compact.x = min(compact.x + accum.x, max_compaction);
    compact.y += accum.y;
    compact.z += accum.z;
    compact.w += accum.w;

    pos_out[i] = p;
    vel_out[i] = v;
    compact_out[i] = compact;
}

@compute @workgroup_size(64)
fn apply_feedback_in_place(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.dims.x) {
        return;
    }
    let mass = max(u.params.x, 1.0e-9);
    let dt = max(u.params.y, 0.0);
    let gain = max(u.params.z, 0.0);
    let max_compaction = max(u.params.w, 0.0);
    let damping = max(u.params2.x, 0.0);

    var p = pos_rw[i];
    var v = vel_rw[i];
    var compact = compact_rw[i];
    var dv = vec3<f32>(0.0, 0.0, 0.0);
    var accum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var damping_weight = 0.0;

    var c = 0u;
    loop {
        if (c >= u.dims.y) {
            break;
        }
        let contact = contacts_in_place[c];
        let influence = max(contact.center_radius.w, 1.0e-6);
        let dx = p.x - contact.center_radius.x;
        let dz = p.z - contact.center_radius.z;
        let horizontal = sqrt(dx * dx + dz * dz);
        if (horizontal <= influence) {
            let below_surface = contact.center_radius.y - p.y;
            if (below_surface >= -influence * 0.5 && below_surface <= influence * 2.0) {
                let radial_w = clamp(1.0 - horizontal / influence, 0.0, 1.0);
                let depth_w = clamp(1.0 - max(below_surface, 0.0) / (influence * 2.0), 0.10, 1.0);
                let w = radial_w * radial_w * depth_w;
                let shear = length(vec2<f32>(contact.force_pressure.x, contact.force_pressure.z));
                let pressure = max(contact.force_pressure.w, max(-contact.force_pressure.y, 0.0));
                dv += contact.force_pressure.xyz * (w * dt / mass);
                accum.x += pressure * w * gain * dt;
                accum.y += max(-contact.force_pressure.y, 0.0) * w * dt;
                accum.z += shear * w * dt;
                accum.w += w;
                damping_weight = max(damping_weight, w);
            }
        }
        c += 1u;
    }

    let damp = clamp(damping * damping_weight * dt, 0.0, 0.95);
    v.x = (v.x + dv.x) * (1.0 - damp);
    v.y = (v.y + dv.y) * (1.0 - damp);
    v.z = (v.z + dv.z) * (1.0 - damp);
    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;
    compact.x = min(compact.x + accum.x, max_compaction);
    compact.y += accum.y;
    compact.z += accum.z;
    compact.w += accum.w;

    pos_rw[i] = p;
    vel_rw[i] = v;
    compact_rw[i] = compact;
}

@compute @workgroup_size(64)
fn raster_terrain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let particle_count = ru.dims.x;
    let nx = ru.dims.y;
    let nz = ru.dims.z;
    let total = nx * nz;
    if (idx >= total) {
        return;
    }
    let ix = idx % nx;
    let iz = idx / nx;
    let origin_x = ru.terrain0.x;
    let origin_z = ru.terrain0.y;
    let spacing = max(ru.terrain0.z, 1.0e-6);
    let rest_height = ru.terrain0.w;

    var sink = 0.0;
    var h = rest_height;
    var vel_sum = vec3<f32>(0.0, 0.0, 0.0);
    var vel_count = 0.0;
    var pidx = 0u;
    loop {
        if (pidx >= particle_count) {
            break;
        }
        let p = raster_pos[pidx];
        let c = raster_compact[pidx];
        let pix_raw = i32(floor((p.x - origin_x) / spacing));
        let piz_raw = i32(floor((p.z - origin_z) / spacing));
        let pix = u32(clamp(pix_raw, 0, i32(nx) - 1));
        let piz = u32(clamp(piz_raw, 0, i32(nz) - 1));
        if (pix == ix && piz == iz) {
            let compact = max(c.x, 0.0);
            sink = max(sink, compact);
            h = min(h, rest_height - compact);
            vel_sum += raster_vel[pidx].xyz;
            vel_count += 1.0;
        }
        pidx += 1u;
    }
    raster_heights[idx] = h;
    raster_sinkage[idx] = sink;
    if (vel_count > 0.0) {
        raster_surface_velocity[idx] = vec4<f32>(vel_sum / vel_count, 0.0);
    } else {
        raster_surface_velocity[idx] = vec4<f32>(ru.terrain_velocity.xyz, 0.0);
    }
}
