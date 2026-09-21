struct SpatialVelocity { linear: vec4<f32>, angular: vec4<f32> }
struct Params { particles: u32, bodies: u32, phase: u32, dofs: u32 }
@group(0) @binding(0) var<storage, read> particle_mass: array<f32>;
@group(0) @binding(1) var<storage, read> particle_velocity: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> body_mass: array<f32>;
@group(0) @binding(3) var<storage, read> body_velocity: array<SpatialVelocity>;
@group(0) @binding(4) var<storage, read_write> diagnostics: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read> joint_velocity: array<f32>;
@group(0) @binding(7) var<storage, read> jacobians: array<f32>;
var<workgroup> soft_sum: array<vec4<f32>, 64>;
var<workgroup> rigid_sum: array<vec4<f32>, 64>;

@compute @workgroup_size(64)
fn record(@builtin(local_invocation_index) lane: u32) {
    var soft = vec4<f32>(0.0);
    var compensation = vec4<f32>(0.0);
    for (var i = lane; i < params.particles; i += 64u) {
        let value = vec4<f32>(particle_velocity[i].xyz * particle_mass[i], 0.0) - compensation;
        let next = soft + value;
        compensation = (next - soft) - value;
        soft = next;
    }
    var rigid = vec4<f32>(0.0);
    for (var i = lane; i < params.bodies; i += 64u) {
        var linear = body_velocity[i].linear.xyz;
        if params.dofs > 0u {
            linear = vec3<f32>(0.0);
            for (var d = 0u; d < params.dofs; d += 1u) {
                let index = i*6u*params.dofs+d;
                linear += vec3<f32>(jacobians[index], jacobians[index+params.dofs], jacobians[index+2u*params.dofs]) * joint_velocity[d];
            }
        }
        rigid += vec4<f32>(linear * body_mass[i], 0.0);
    }
    soft_sum[lane] = soft;
    rigid_sum[lane] = rigid;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride {
            soft_sum[lane] += soft_sum[lane+stride];
            rigid_sum[lane] += rigid_sum[lane+stride];
        }
        workgroupBarrier();
    }
    if lane == 0u {
        let slot = params.phase * 2u;
        if params.phase > 0u {
            diagnostics[10u+slot] += soft_sum[0] - diagnostics[slot-2u];
            diagnostics[11u+slot] += rigid_sum[0] - diagnostics[slot-1u];
        }
        diagnostics[slot] = soft_sum[0];
        diagnostics[slot+1u] = rigid_sum[0];
    }
}
