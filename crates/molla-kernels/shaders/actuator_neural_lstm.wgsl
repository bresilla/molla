// Stage-31 single-cell LSTM forward pass + linear projection.
//
// Architecture (all sizes runtime-configurable via the uniform block):
//   x_t     : [input_dim]
//   W_*     : [hidden_dim × input_dim]   (4 gates: i, f, g, o)
//   U_*     : [hidden_dim × hidden_dim]  (4 gates)
//   b_*     : [hidden_dim]               (4 gates)
//   h_{t-1} : [hidden_dim]   ← persistent across substeps (host owns the
//                              buffer; we just read+write in place).
//   c_{t-1} : [hidden_dim]   ← idem.
//
// Recurrence:
//   i_t = sigmoid(W_i · x_t + U_i · h_{t-1} + b_i)
//   f_t = sigmoid(W_f · x_t + U_f · h_{t-1} + b_f)
//   g_t = tanh   (W_g · x_t + U_g · h_{t-1} + b_g)
//   o_t = sigmoid(W_o · x_t + U_o · h_{t-1} + b_o)
//   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
//   h_t = o_t ⊙ tanh(c_t)
//   y_t = W_y · h_t + b_y               ← linear projection to outputs
//
// Single workgroup, 64 threads. Hidden state buffers (h, c) are
// `read_write` storage so we can both consume the previous-step values
// and write the next-step values in the same pass — using the standard
// "stage to local, barrier, write" pattern to avoid intra-step races.
//
// Weights stored row-major: `m[row, col] = m[row * cols + col]`.

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
@group(0) @binding(2) var<storage, read_write> h_state: array<f32>;
@group(0) @binding(3) var<storage, read_write> c_state: array<f32>;

@group(1) @binding(0) var<storage, read> w_i: array<f32>;
@group(1) @binding(1) var<storage, read> u_i: array<f32>;
@group(1) @binding(2) var<storage, read> b_i: array<f32>;
@group(1) @binding(3) var<storage, read> w_f: array<f32>;
@group(1) @binding(4) var<storage, read> u_f: array<f32>;
@group(1) @binding(5) var<storage, read> b_f: array<f32>;
@group(1) @binding(6) var<storage, read> w_g: array<f32>;
@group(1) @binding(7) var<storage, read> u_g: array<f32>;
@group(1) @binding(8) var<storage, read> b_g: array<f32>;

@group(2) @binding(0) var<storage, read> w_o: array<f32>;
@group(2) @binding(1) var<storage, read> u_o: array<f32>;
@group(2) @binding(2) var<storage, read> b_o: array<f32>;
@group(2) @binding(3) var<storage, read> w_y: array<f32>;
@group(2) @binding(4) var<storage, read> b_y: array<f32>;

@group(3) @binding(0) var<uniform> u: Uniforms;

// Workgroup scratch — staged previous-step h/c so all threads see a
// consistent snapshot during the gate accumulation phase, plus the
// freshly-computed h_t for the linear-projection phase.
var<workgroup> h_prev: array<f32, 256>;
var<workgroup> c_prev: array<f32, 256>;
var<workgroup> h_new:  array<f32, 256>;

fn sigmoidf(z: f32) -> f32 {
    // Numerically stable sigmoid for both signs of z.
    if (z >= 0.0) {
        let e = exp(-z);
        return 1.0 / (1.0 + e);
    } else {
        let e = exp(z);
        return e / (1.0 + e);
    }
}

@compute @workgroup_size(64)
fn forward(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let in_dim  = u.input_dim;
    let hid_dim = u.hidden_dim;
    let out_dim = u.output_dim;

    // ── Stage prev-step state into workgroup scratch ────────────────
    var s: u32 = tid;
    loop {
        if (s >= hid_dim) { break; }
        if (s < MAX_HIDDEN) {
            h_prev[s] = h_state[s];
            c_prev[s] = c_state[s];
        }
        s = s + WG;
    }
    workgroupBarrier();

    // ── Compute h_t / c_t per hidden unit ───────────────────────────
    var i: u32 = tid;
    loop {
        if (i >= hid_dim) { break; }
        if (i < MAX_HIDDEN) {
            // Pre-activations for the four gates.
            var z_i: f32 = b_i[i];
            var z_f: f32 = b_f[i];
            var z_g: f32 = b_g[i];
            var z_o: f32 = b_o[i];
            for (var k: u32 = 0u; k < in_dim; k = k + 1u) {
                let xk = inputs[k];
                z_i = z_i + w_i[i * in_dim + k] * xk;
                z_f = z_f + w_f[i * in_dim + k] * xk;
                z_g = z_g + w_g[i * in_dim + k] * xk;
                z_o = z_o + w_o[i * in_dim + k] * xk;
            }
            for (var k: u32 = 0u; k < hid_dim; k = k + 1u) {
                let hk = h_prev[k];
                z_i = z_i + u_i[i * hid_dim + k] * hk;
                z_f = z_f + u_f[i * hid_dim + k] * hk;
                z_g = z_g + u_g[i * hid_dim + k] * hk;
                z_o = z_o + u_o[i * hid_dim + k] * hk;
            }
            let it = sigmoidf(z_i);
            let ft = sigmoidf(z_f);
            let gt = tanh(z_g);
            let ot = sigmoidf(z_o);
            let ct = ft * c_prev[i] + it * gt;
            let ht = ot * tanh(ct);
            // Persist next-step state back to global memory.
            c_state[i] = ct;
            h_state[i] = ht;
            h_new[i]   = ht;
        }
        i = i + WG;
    }
    workgroupBarrier();

    // ── Linear projection: y = W_y · h_t + b_y ──────────────────────
    var o: u32 = tid;
    loop {
        if (o >= out_dim) { break; }
        var acc: f32 = b_y[o];
        for (var k: u32 = 0u; k < hid_dim; k = k + 1u) {
            acc = acc + w_y[o * hid_dim + k] * h_new[k];
        }
        outputs[o] = acc;
        o = o + WG;
    }
}
