// Scatter pass for `ControllerNeuralMlp`.
//
// One thread per network output: `joint_force[dof_indices[i]] = outputs[i]`.
// Lives in its own shader so the forward pass stays a pure compute
// primitive (reusable for off-pipeline neural inference too).

struct Uniforms {
    output_dim: u32,
    _pad0:      u32,
    _pad1:      u32,
    _pad2:      u32,
};

@group(0) @binding(0) var<storage, read>       outputs:     array<f32>;
@group(0) @binding(1) var<storage, read>       dof_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> joint_force: array<f32>;

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.output_dim) { return; }
    let dof = dof_indices[i];
    joint_force[dof] = outputs[i];
}
