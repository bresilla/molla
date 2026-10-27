// Fixed-architecture 3-layer MLP forward pass — Stage 18.1.
//
// Architecture: input → hidden1 → hidden2 → output, with tanh
// non-linearities on the hidden layers and a linear output. Sizes
// are passed in via the uniform block; the same shader handles any
// (input_dim, hidden_dim, output_dim) at runtime.
//
// Layout (one workgroup per network instance — single-batch, single
// policy; we don't need batched inference for the on-device control
// loop):
//
//   layer 1: w1 [hidden_dim × input_dim]  + b1 [hidden_dim]   → tanh
//   layer 2: w2 [hidden_dim × hidden_dim] + b2 [hidden_dim]   → tanh
//   layer 3: w3 [output_dim × hidden_dim] + b3 [output_dim]   → linear
//
// Weights are stored row-major: `w[row, col] = weights[row * cols + col]`.
//
// We dispatch a single workgroup of `WG = 64` threads. Each phase has
// a workgroup-barrier between it and the next, so layer N reads the
// completed activation buffer for layer N-1.
//
// `forward` writes the raw network outputs into `outputs[0..output_dim]`.
// `scatter_to_joint_force` is a separate one-thread-per-output entry
// point that copies `outputs[i]` into `joint_force[dof_indices[i]]`,
// so the MLP can drop straight into the actuator pipeline.

const WG: u32 = 64u;
const MAX_HIDDEN: u32 = 256u;

struct Uniforms {
    input_dim:  u32,
    hidden_dim: u32,
    output_dim: u32,
    _pad0:      u32,
};

@group(0) @binding(0) var<storage, read>       inputs:  array<f32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;

@group(1) @binding(0) var<storage, read> w1: array<f32>;
@group(1) @binding(1) var<storage, read> b1: array<f32>;
@group(1) @binding(2) var<storage, read> w2: array<f32>;
@group(1) @binding(3) var<storage, read> b2: array<f32>;
@group(1) @binding(4) var<storage, read> w3: array<f32>;
@group(1) @binding(5) var<storage, read> b3: array<f32>;

@group(2) @binding(0) var<uniform> u: Uniforms;

// Scratch per-workgroup activations. Sized for hidden_dim ≤ 256, which
// covers the demos in this iteration; if a future user needs wider
// hidden layers, bump this constant + add validation in the host.
var<workgroup> h1: array<f32, 256>;
var<workgroup> h2: array<f32, 256>;

@compute @workgroup_size(64)
fn forward(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let in_dim  = u.input_dim;
    let hid_dim = u.hidden_dim;
    let out_dim = u.output_dim;

    // ── Layer 1: input → hidden1 (tanh) ─────────────────────────────
    var i: u32 = tid;
    loop {
        if (i >= hid_dim) { break; }
        if (i < MAX_HIDDEN) {
            var acc: f32 = b1[i];
            for (var k: u32 = 0u; k < in_dim; k = k + 1u) {
                acc = acc + w1[i * in_dim + k] * inputs[k];
            }
            h1[i] = tanh(acc);
        }
        i = i + WG;
    }
    workgroupBarrier();

    // ── Layer 2: hidden1 → hidden2 (tanh) ───────────────────────────
    var j: u32 = tid;
    loop {
        if (j >= hid_dim) { break; }
        if (j < MAX_HIDDEN) {
            var acc: f32 = b2[j];
            for (var k: u32 = 0u; k < hid_dim; k = k + 1u) {
                acc = acc + w2[j * hid_dim + k] * h1[k];
            }
            h2[j] = tanh(acc);
        }
        j = j + WG;
    }
    workgroupBarrier();

    // ── Layer 3: hidden2 → output (linear) ──────────────────────────
    var o: u32 = tid;
    loop {
        if (o >= out_dim) { break; }
        var acc: f32 = b3[o];
        for (var k: u32 = 0u; k < hid_dim; k = k + 1u) {
            acc = acc + w3[o * hid_dim + k] * h2[k];
        }
        outputs[o] = acc;
        o = o + WG;
    }
}
