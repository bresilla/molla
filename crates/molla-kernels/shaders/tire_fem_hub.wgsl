// Stage (f) — tire FEM hub reduction + 3-DOF hub integrator.
//
// One workgroup reduces the bead reactions on all three axes and
// integrates the full 3-DOF hub motion (X/Y/Z translation). The
// CPU supplies an external impulse vector each substep
// (`external_force_xyz`) so vehicle-side controllers can apply
// steering/drive/lateral forces directly to the hub. The kernel
// writes back the updated `hub_state` storage buffer that the
// next `step_nodes` dispatch reads as its hub.
//
// Two entry points:
//  - `hub_reduce` — bead reduction + integration; any contact wrench
//    arrives pre-summed on the host via `external_force_xyz`.
//  - `hub_reduce_with_tread_wrench` — additionally reduces the per-node
//    contact-feedback force rows (`tread_force_in`, row count in
//    `load.z`) on-device and feeds their X/Z sum into the hub force, so
//    the rolling propulsion never round-trips through the host. The Y
//    component is deliberately ignored: the normal load reaches the hub
//    through the mesh + bead reduction, matching the CPU `step_inner`
//    convention where only `net_contact_fx/fz` integrate `hub_v_xz`.
//
// `hub_state[0] = (hub_x, hub_y, hub_z, hub_v_y)`
// `hub_state[1] = (hub_v_x, hub_v_z, hub_omega, hub_angle)`
//
// Spin DOF (F2a): when `spin.w > 0.5` the kernel integrates the wheel
// spin on-device, mirroring `TetraTireFem::integrate_spin_dof`
// exactly: `ω += (drive − brake·soft_sign(ω))/I · dt; θ += ω·dt` with
// the ω/(|ω|+0.5) soft-sign saturator and NaN guards. When the flag is
// 0 the zw slots pass through untouched (bit-compatible with the
// pre-F2a host-integrated spin).

const WG: u32 = 128u;

struct HubUniforms {
    // x=bead_count, y=hub_floor_active (1.0/0.0), z=hub_floor_y, w=dt
    cfg: vec4<f32>,
    // x=bead_stiffness, y=bead_damping, z=hub_mass, w=hub_damping
    mech: vec4<f32>,
    // x=downforce, y=gravity_y, z=tread force row count (with_tread entry), w=reserved
    load: vec4<f32>,
    // xyz=external hub force (N), w=lateral damping coefficient
    extf: vec4<f32>,
    // x=drive torque (N·m), y=brake torque (N·m, ≥0),
    // z=wheel spin inertia (kg·m²), w=spin-DOF active flag (1.0/0.0)
    spin: vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       bead_ids:     array<u32>;
@group(0) @binding(1) var<storage, read>       bead_offsets: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       pos_in:       array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       vel_in:       array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> hub_state:    array<vec4<f32>>;
// Only referenced by `hub_reduce_with_tread_wrench`; the plain
// `hub_reduce` pipeline layout does not need to bind it.
@group(0) @binding(5) var<storage, read>       tread_force_in: array<vec4<f32>>;
@group(1) @binding(0) var<uniform>             h:            HubUniforms;

var<workgroup> partial_fx: array<f32, WG>;
var<workgroup> partial_fy: array<f32, WG>;
var<workgroup> partial_fz: array<f32, WG>;

// Bead reduction + 3-DOF hub integration shared by both entry points.
// `extra_force` is added to the hub force on top of bead reactions,
// gravity, downforce, and the uniform external force; every thread must
// pass the same value.
fn integrate_hub(i: u32, extra_force: vec3<f32>) {
    let bead_count = u32(h.cfg.x);
    let hub = hub_state[0u];          // (hub_x, hub_y, hub_z, hub_v_y)
    let hub_v_xz = hub_state[1u];      // (hub_v_x, hub_v_z, _, _)
    let bead_k = h.mech.x;
    let bead_c = h.mech.y;

    var local_x: f32 = 0.0;
    var local_y: f32 = 0.0;
    var local_z: f32 = 0.0;
    if (i < bead_count) {
        let id = bead_ids[i];
        let offset = bead_offsets[i].xyz;
        let pos = pos_in[id].xyz;
        let vel = vel_in[id].xyz;
        let bead_target = vec3<f32>(hub.x, hub.y, hub.z) + offset;
        let delta = bead_target - pos;
        let rel_v = vec3<f32>(hub_v_xz.x, hub.w, hub_v_xz.y) - vel;
        let force_on_bead = bead_k * delta + bead_c * rel_v;
        local_x = force_on_bead.x;
        local_y = force_on_bead.y;
        local_z = force_on_bead.z;
    }
    partial_fx[i] = local_x;
    partial_fy[i] = local_y;
    partial_fz[i] = local_z;
    workgroupBarrier();

    // Power-of-two tree reduction across all three axes.
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (i < stride) {
            partial_fx[i] = partial_fx[i] + partial_fx[i + stride];
            partial_fy[i] = partial_fy[i] + partial_fy[i + stride];
            partial_fz[i] = partial_fz[i] + partial_fz[i + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (i == 0u) {
        let downforce = h.load.x;
        let gravity_y = h.load.y;
        let hub_mass = h.mech.z;
        let hub_damp = h.mech.w;
        let lat_damp = h.extf.w;
        let dt = h.cfg.w;

        // Bead reactions are equal and opposite to the forces we just
        // summed (forces ON the bead).
        let reaction_x = -partial_fx[0];
        let reaction_y = -partial_fy[0];
        let reaction_z = -partial_fz[0];

        // 3-DOF hub force = bead reactions + downforce + gravity + steer/drive.
        var force_x = reaction_x + h.extf.x + extra_force.x;
        var force_y = reaction_y - downforce + gravity_y * hub_mass + h.extf.y + extra_force.y;
        var force_z = reaction_z + h.extf.z + extra_force.z;

        // Implicit damping for unconditional stability.
        let denom_y = hub_mass + hub_damp * dt;
        var v_new_y = (hub_mass * hub.w + force_y * dt) / denom_y;
        v_new_y = clamp(v_new_y, -2.5, 2.5);

        let denom_xz = hub_mass + lat_damp * dt;
        var v_new_x = (hub_mass * hub_v_xz.x + force_x * dt) / denom_xz;
        var v_new_z = (hub_mass * hub_v_xz.y + force_z * dt) / denom_xz;
        v_new_x = clamp(v_new_x, -25.0, 25.0);
        v_new_z = clamp(v_new_z, -25.0, 25.0);

        var x_new = hub.x + v_new_x * dt;
        var y_new = hub.y + v_new_y * dt;
        var z_new = hub.z + v_new_z * dt;

        // Hub floor (rim cannot sink through the ground).
        if (h.cfg.y > 0.5) {
            let hub_floor = h.cfg.z;
            if (y_new < hub_floor) {
                y_new = hub_floor;
                if (v_new_y < 0.0) { v_new_y = 0.0; }
            }
        }
        hub_state[0u] = vec4<f32>(x_new, y_new, z_new, v_new_y);

        // Spin DOF — resident when enabled, passthrough otherwise.
        var omega = hub_v_xz.z;
        var angle = hub_v_xz.w;
        if (h.spin.w > 0.5) {
            let inertia = max(h.spin.z, 1.0e-6);
            let brake_soft = omega / (abs(omega) + 0.5);
            let alpha = (h.spin.x - h.spin.y * brake_soft) / inertia;
            omega = omega + alpha * dt;
            if (!(omega == omega)) { omega = 0.0; }
            angle = angle + omega * dt;
            if (!(angle == angle)) { angle = 0.0; }
        }
        hub_state[1u] = vec4<f32>(v_new_x, v_new_z, omega, angle);
    }
}

@compute @workgroup_size(WG)
fn hub_reduce(@builtin(local_invocation_id) lid: vec3<u32>) {
    integrate_hub(lid.x, vec3<f32>(0.0));
}

@compute @workgroup_size(WG)
fn hub_reduce_with_tread_wrench(@builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lid.x;
    // Strided reduction of the contact-feedback force rows (one row per
    // FEM node; non-contact rows are zero). X/Z only — see header.
    let rows = u32(h.load.z);
    var sum_x: f32 = 0.0;
    var sum_z: f32 = 0.0;
    var idx = i;
    loop {
        if (idx >= rows) { break; }
        sum_x = sum_x + tread_force_in[idx].x;
        sum_z = sum_z + tread_force_in[idx].z;
        idx = idx + WG;
    }
    partial_fx[i] = sum_x;
    partial_fy[i] = 0.0;
    partial_fz[i] = sum_z;
    workgroupBarrier();
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (i < stride) {
            partial_fx[i] = partial_fx[i] + partial_fx[i + stride];
            partial_fz[i] = partial_fz[i] + partial_fz[i + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let wrench = vec3<f32>(partial_fx[0], 0.0, partial_fz[0]);
    // The bead phase below reuses the partial arrays — make sure every
    // thread has read the wrench before they are overwritten.
    workgroupBarrier();
    integrate_hub(i, wrench);
}
