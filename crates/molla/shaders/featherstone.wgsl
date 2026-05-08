// Featherstone-style forward dynamics on GPU — single-chain
// revolute-joint version mirroring `solvers::featherstone::cpu_step`.
//
// Parallelism strategy: one thread per articulation. For molla's
// stage-3.x demos there's typically one articulation per scene
// (one robot), so this is effectively single-threaded — the GPU
// path mostly exists to avoid host-device sync between physics
// substeps and to align with the project's GPU-first rule.
//
// Each thread walks its articulation's joint range and integrates
// each revolute joint independently using the same per-joint
// approximation as the CPU path. Joint coupling (proper ABA) lands
// in a stage-3.x rewrite once branched chains are supported.

struct Transform {
    position: vec4<f32>, // xyz + pad
    rotation: vec4<f32>, // (qx, qy, qz, qw)
};

struct Mat3Cols {
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
};

struct Uniforms {
    dt:                 f32,
    joint_damping:      f32,
    joint_count:        u32,
    articulation_count: u32,
};

const JOINT_PRISMATIC: u32 = 0u;
const JOINT_REVOLUTE:  u32 = 1u;
const JOINT_FIXED:     u32 = 3u;
const JOINT_FREE:      u32 = 4u;

// Group 0 — Model
@group(0) @binding(0) var<storage, read> joint_type:         array<u32>;
@group(0) @binding(1) var<storage, read> joint_parent:       array<i32>;
@group(0) @binding(2) var<storage, read> joint_child:        array<i32>;
@group(0) @binding(3) var<storage, read> joint_x_p:          array<Transform>;
@group(0) @binding(4) var<storage, read> joint_x_c:          array<Transform>;
@group(0) @binding(5) var<storage, read> joint_axis:         array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> joint_coord_offset: array<u32>;
@group(0) @binding(7) var<storage, read> joint_dof_offset:   array<u32>;

// Group 1 — Body data + State
@group(1) @binding(0) var<storage, read>       body_mass:    array<f32>;
@group(1) @binding(1) var<storage, read>       body_inertia: array<Mat3Cols>;
@group(1) @binding(2) var<storage, read>       body_com:     array<vec4<f32>>;
@group(1) @binding(3) var<storage, read>       body_world:   array<i32>;
@group(1) @binding(4) var<storage, read>       gravity:      array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> body_q:       array<Transform>;
@group(1) @binding(6) var<storage, read_write> joint_q:      array<f32>;
@group(1) @binding(7) var<storage, read_write> joint_qd:     array<f32>;

// Group 2 — Control
@group(2) @binding(0) var<storage, read> joint_force: array<f32>;

// Group 3 — Articulation ranges. Each articulation `a` owns joints
// `art_offsets[a] .. art_offsets[a+1]`. The CSR is uploaded once at
// construction and read every step.
@group(3) @binding(0) var<storage, read> art_offsets: array<u32>;

// Group 4 — Uniforms
@group(4) @binding(0) var<uniform> uniforms: Uniforms;

// ─── Helpers ─────────────────────────────────────────────────────────

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
    );
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s*s - dot(u,u)) * v + 2.0 * s * cross(u, v);
}

fn quat_from_axis_angle(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    let h = 0.5 * angle;
    let s = sin(h);
    return vec4<f32>(axis * s, cos(h));
}

fn transform_compose(a: Transform, b: Transform) -> Transform {
    var r: Transform;
    let ax = a.rotation;
    let bx = b.rotation;
    let bp = b.position.xyz;
    r.rotation = quat_mul(ax, bx);
    r.rotation = normalize(r.rotation);
    r.position = vec4<f32>(a.position.xyz + quat_rotate(ax, bp), 0.0);
    return r;
}

fn transform_inverse(t: Transform) -> Transform {
    var r: Transform;
    r.rotation = quat_conj(t.rotation);
    r.position = vec4<f32>(-quat_rotate(r.rotation, t.position.xyz), 0.0);
    return r;
}

fn mat3_from_cols(m: Mat3Cols) -> mat3x3<f32> {
    return mat3x3<f32>(m.col0.xyz, m.col1.xyz, m.col2.xyz);
}

fn joint_local_transform(jt: u32, q_start: u32, axis: vec3<f32>) -> Transform {
    var t: Transform;
    t.position = vec4<f32>(0.0);
    t.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (jt == JOINT_REVOLUTE) {
        t.rotation = quat_from_axis_angle(axis, joint_q[q_start]);
    } else if (jt == JOINT_PRISMATIC) {
        t.position = vec4<f32>(axis * joint_q[q_start], 0.0);
    } else if (jt == JOINT_FREE) {
        // 7 coords: tx, ty, tz, qx, qy, qz, qw
        t.position = vec4<f32>(joint_q[q_start], joint_q[q_start+1u], joint_q[q_start+2u], 0.0);
        t.rotation = normalize(vec4<f32>(joint_q[q_start+3u], joint_q[q_start+4u], joint_q[q_start+5u], joint_q[q_start+6u]));
    }
    return t;
}

fn inertia_about_axis_world(rotation: vec4<f32>, i_local: mat3x3<f32>, axis_world: vec3<f32>) -> f32 {
    let axis_local = quat_rotate(quat_conj(rotation), axis_world);
    return dot(axis_world, quat_rotate(rotation, i_local * axis_local));
}

// ─── Kernel ──────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn featherstone_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let art = gid.x;
    if (art >= uniforms.articulation_count) {
        return;
    }
    let j_begin = art_offsets[art];
    let j_end   = art_offsets[art + 1u];

    // ── Pass 1: forward kinematics over this articulation ────────
    for (var j = j_begin; j < j_end; j = j + 1u) {
        let parent = joint_parent[j];
        let child  = u32(joint_child[j]);
        let coord  = joint_coord_offset[j];
        let jt     = joint_type[j];
        let axis   = joint_axis[j].xyz;

        var parent_world: Transform;
        if (parent < 0) {
            parent_world.position = vec4<f32>(0.0);
            parent_world.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        } else {
            parent_world = body_q[u32(parent)];
        }

        let local = joint_local_transform(jt, coord, axis);
        let xp = joint_x_p[j];
        let xc_inv = transform_inverse(joint_x_c[j]);
        let xpw = transform_compose(parent_world, xp);
        let after = transform_compose(xpw, local);
        body_q[child] = transform_compose(after, xc_inv);
    }

    // ── Pass 2: per-joint dynamics + integrate ───────────────────
    for (var j = j_begin; j < j_end; j = j + 1u) {
        if (joint_type[j] != JOINT_REVOLUTE) {
            continue;
        }
        let parent = joint_parent[j];
        let child  = u32(joint_child[j]);
        var parent_world: Transform;
        if (parent < 0) {
            parent_world.position = vec4<f32>(0.0);
            parent_world.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        } else {
            parent_world = body_q[u32(parent)];
        }
        let joint_world = transform_compose(parent_world, joint_x_p[j]);
        let axis_world = quat_rotate(joint_world.rotation, joint_axis[j].xyz);

        let cw = body_q[child];
        let com_world = cw.position.xyz + quat_rotate(cw.rotation, body_com[child].xyz);
        let r = com_world - joint_world.position.xyz;

        let world_idx = max(body_world[child], 0);
        let g = gravity[u32(world_idx)].xyz;
        let grav_force = g * body_mass[child];
        let tau_grav = dot(cross(r, grav_force), axis_world);

        let dof_idx = joint_dof_offset[j];
        let tau_applied = joint_force[dof_idx];

        let i_local = mat3_from_cols(body_inertia[child]);
        let inertia_axis = inertia_about_axis_world(cw.rotation, i_local, axis_world);
        let i_eff = inertia_axis + body_mass[child] * dot(r, r);
        if (i_eff <= 0.0) {
            continue;
        }
        let q_dd = (tau_applied + tau_grav) / i_eff;

        let qd_old = joint_qd[dof_idx];
        let qd_int = qd_old + q_dd * uniforms.dt;
        let damp = max(1.0 - uniforms.joint_damping * uniforms.dt, 0.0);
        let qd_new = qd_int * damp;
        joint_qd[dof_idx] = qd_new;

        let coord_idx = joint_coord_offset[j];
        joint_q[coord_idx] = joint_q[coord_idx] + qd_new * uniforms.dt;
    }

    // ── Pass 3: re-evaluate FK so body_q reflects integrated q ───
    for (var j = j_begin; j < j_end; j = j + 1u) {
        let parent = joint_parent[j];
        let child  = u32(joint_child[j]);
        let coord  = joint_coord_offset[j];
        let jt     = joint_type[j];
        let axis   = joint_axis[j].xyz;

        var parent_world: Transform;
        if (parent < 0) {
            parent_world.position = vec4<f32>(0.0);
            parent_world.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        } else {
            parent_world = body_q[u32(parent)];
        }

        let local = joint_local_transform(jt, coord, axis);
        let xp = joint_x_p[j];
        let xc_inv = transform_inverse(joint_x_c[j]);
        let xpw = transform_compose(parent_world, xp);
        let after = transform_compose(xpw, local);
        body_q[child] = transform_compose(after, xc_inv);
    }
}
