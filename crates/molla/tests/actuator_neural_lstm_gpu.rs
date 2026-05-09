//! Stage 31 — `ControllerNeuralLstm` GPU forward-pass + hidden-state
//! persistence tests. `#[ignore]` because they need a real wgpu
//! adapter; run with `nixVulkan cargo test -p molla --features gpu \
//! --test actuator_neural_lstm_gpu -- --ignored`.

#![cfg(feature = "gpu")]

use molla::actuators::{
    ControllerNeuralLstm, ControllerNeuralLstmConfig, LstmWeights,
};
use molla::compute::Engine;
use molla::math::Real;

/// CPU oracle: single-step LSTM forward pass + linear projection.
/// Returns `(h_t, c_t, y_t)`.
fn cpu_lstm_step(
    x: &[Real],
    h_prev: &[Real],
    c_prev: &[Real],
    weights: &LstmWeights,
    in_dim: usize,
    hid_dim: usize,
    out_dim: usize,
) -> (Vec<Real>, Vec<Real>, Vec<Real>) {
    let sigmoid = |z: Real| 1.0 / (1.0 + (-z).exp());

    let mut h_t = vec![0.0 as Real; hid_dim];
    let mut c_t = vec![0.0 as Real; hid_dim];

    for i in 0..hid_dim {
        let mut z_i = weights.b_i[i];
        let mut z_f = weights.b_f[i];
        let mut z_g = weights.b_g[i];
        let mut z_o = weights.b_o[i];
        for k in 0..in_dim {
            z_i += weights.w_i[i * in_dim + k] * x[k];
            z_f += weights.w_f[i * in_dim + k] * x[k];
            z_g += weights.w_g[i * in_dim + k] * x[k];
            z_o += weights.w_o[i * in_dim + k] * x[k];
        }
        for k in 0..hid_dim {
            z_i += weights.u_i[i * hid_dim + k] * h_prev[k];
            z_f += weights.u_f[i * hid_dim + k] * h_prev[k];
            z_g += weights.u_g[i * hid_dim + k] * h_prev[k];
            z_o += weights.u_o[i * hid_dim + k] * h_prev[k];
        }
        let it = sigmoid(z_i);
        let ft = sigmoid(z_f);
        let gt = z_g.tanh();
        let ot = sigmoid(z_o);
        let ct = ft * c_prev[i] + it * gt;
        let ht = ot * ct.tanh();
        c_t[i] = ct;
        h_t[i] = ht;
    }

    let mut y = vec![0.0 as Real; out_dim];
    for o in 0..out_dim {
        let mut acc = weights.b_y[o];
        for k in 0..hid_dim {
            acc += weights.w_y[o * hid_dim + k] * h_t[k];
        }
        y[o] = acc;
    }
    (h_t, c_t, y)
}

/// Build a small known-weight LSTM (3 → 4 → 2) with mildly non-trivial
/// values so `h` and `c` actually evolve and the output isn't a
/// constant.
struct KnownNet {
    w_i: Vec<Real>, u_i: Vec<Real>, b_i: Vec<Real>,
    w_f: Vec<Real>, u_f: Vec<Real>, b_f: Vec<Real>,
    w_g: Vec<Real>, u_g: Vec<Real>, b_g: Vec<Real>,
    w_o: Vec<Real>, u_o: Vec<Real>, b_o: Vec<Real>,
    w_y: Vec<Real>, b_y: Vec<Real>,
    in_dim: usize, hid_dim: usize, out_dim: usize,
}

fn build_known_net() -> KnownNet {
    let in_dim = 3usize;
    let hid_dim = 4usize;
    let out_dim = 2usize;

    // Deterministic, easy-to-debug values. The shapes are tiny so a
    // mismatch is easy to spot in the diagnostic prints.
    let mut w_i = vec![0.0; hid_dim * in_dim];
    let mut w_f = vec![0.0; hid_dim * in_dim];
    let mut w_g = vec![0.0; hid_dim * in_dim];
    let mut w_o = vec![0.0; hid_dim * in_dim];
    for i in 0..hid_dim {
        for k in 0..in_dim {
            let idx = i * in_dim + k;
            // Spread small weights over the gates so each gate sees
            // a different linear combination of x.
            w_i[idx] = 0.10 * ((i + k + 1) as Real);
            w_f[idx] = 0.05 * ((i * 2 + k) as Real);
            w_g[idx] = 0.20 * ((i + 1) as Real) - 0.10 * (k as Real);
            w_o[idx] = 0.15 * ((i + k) as Real) - 0.05;
        }
    }
    let mut u_i = vec![0.0; hid_dim * hid_dim];
    let mut u_f = vec![0.0; hid_dim * hid_dim];
    let mut u_g = vec![0.0; hid_dim * hid_dim];
    let mut u_o = vec![0.0; hid_dim * hid_dim];
    // Recurrent weights: small off-diagonal coupling + diagonal bias
    // toward "remember the previous hidden value".
    for i in 0..hid_dim {
        for k in 0..hid_dim {
            let idx = i * hid_dim + k;
            u_i[idx] = if i == k { 0.10 } else { 0.02 * ((i as i32 - k as i32) as Real) };
            u_f[idx] = if i == k { 0.30 } else { 0.01 };
            u_g[idx] = if i == k { 0.05 } else { 0.0 };
            u_o[idx] = if i == k { 0.15 } else { -0.02 };
        }
    }
    // Forget-gate bias 1.0 — standard LSTM init trick: pushes the
    // forget gate toward "remember" so c can accumulate.
    let b_i = vec![0.0; hid_dim];
    let b_f = vec![1.0; hid_dim];
    let b_g = vec![0.0; hid_dim];
    let b_o = vec![0.0; hid_dim];

    let mut w_y = vec![0.0; out_dim * hid_dim];
    for o in 0..out_dim {
        for k in 0..hid_dim {
            w_y[o * hid_dim + k] = if o == 0 { 1.0 } else { -0.5 } * ((k + 1) as Real) * 0.25;
        }
    }
    let b_y = vec![0.0, 0.5];

    KnownNet {
        w_i, u_i, b_i,
        w_f, u_f, b_f,
        w_g, u_g, b_g,
        w_o, u_o, b_o,
        w_y, b_y,
        in_dim, hid_dim, out_dim,
    }
}

impl KnownNet {
    fn weights(&self) -> LstmWeights<'_> {
        LstmWeights {
            w_i: &self.w_i, u_i: &self.u_i, b_i: &self.b_i,
            w_f: &self.w_f, u_f: &self.u_f, b_f: &self.b_f,
            w_g: &self.w_g, u_g: &self.u_g, b_g: &self.b_g,
            w_o: &self.w_o, u_o: &self.u_o, b_o: &self.b_o,
            w_y: &self.w_y, b_y: &self.b_y,
        }
    }
}

fn make_engine_or_skip() -> Option<Engine> {
    Engine::new_gpu().ok()
}

/// Test 1: single-step forward pass matches the CPU oracle within 1e-4.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn lstm_forward_matches_cpu_oracle() {
    let Some(engine) = make_engine_or_skip() else { return };

    let net = build_known_net();
    let cfg = ControllerNeuralLstmConfig {
        input_dim: net.in_dim as u32,
        hidden_dim: net.hid_dim as u32,
        output_dim: net.out_dim as u32,
    };
    // dof_indices need only be unique; pick 0..out_dim.
    let dof_indices: Vec<u32> = (0..net.out_dim as u32).collect();
    let mut lstm = ControllerNeuralLstm::new(&engine, cfg, &dof_indices).unwrap();

    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    lstm.set_weights(&device, &queue, &net.weights()).unwrap();

    let x = vec![0.5 as Real, -0.3, 0.8];
    lstm.set_inputs(&queue, &x).unwrap();
    lstm.forward(&engine).unwrap();

    let h_prev = vec![0.0 as Real; net.hid_dim];
    let c_prev = vec![0.0 as Real; net.hid_dim];
    let (h_ref, c_ref, y_ref) = cpu_lstm_step(
        &x, &h_prev, &c_prev, &net.weights(),
        net.in_dim, net.hid_dim, net.out_dim,
    );

    let y_gpu = lstm.download_outputs(&device, &queue).unwrap();
    let h_gpu = lstm.download_hidden_state(&device, &queue).unwrap();
    let c_gpu = lstm.download_cell_state(&device, &queue).unwrap();

    eprintln!("CPU h:  {h_ref:?}");
    eprintln!("GPU h:  {h_gpu:?}");
    eprintln!("CPU c:  {c_ref:?}");
    eprintln!("GPU c:  {c_gpu:?}");
    eprintln!("CPU y:  {y_ref:?}");
    eprintln!("GPU y:  {y_gpu:?}");

    let tol = 1.0e-4;
    for i in 0..net.hid_dim {
        assert!((h_gpu[i] - h_ref[i]).abs() < tol,
            "h[{i}] mismatch: gpu={} cpu={}", h_gpu[i], h_ref[i]);
        assert!((c_gpu[i] - c_ref[i]).abs() < tol,
            "c[{i}] mismatch: gpu={} cpu={}", c_gpu[i], c_ref[i]);
    }
    for o in 0..net.out_dim {
        assert!((y_gpu[o] - y_ref[o]).abs() < tol,
            "y[{o}] mismatch: gpu={} cpu={}", y_gpu[o], y_ref[o]);
    }
}

/// Test 2: hidden state survives across substeps. We feed the same
/// input 10 times in a row and verify
///   (a) the GPU rollout matches a 10-step CPU rollout to 1e-4, and
///   (b) `h` after step 10 differs meaningfully from `h` after step 1
///       — i.e. the recurrence isn't getting silently reset.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn lstm_hidden_state_survives_across_substeps() {
    let Some(engine) = make_engine_or_skip() else { return };

    let net = build_known_net();
    let cfg = ControllerNeuralLstmConfig {
        input_dim: net.in_dim as u32,
        hidden_dim: net.hid_dim as u32,
        output_dim: net.out_dim as u32,
    };
    let dof_indices: Vec<u32> = (0..net.out_dim as u32).collect();
    let mut lstm = ControllerNeuralLstm::new(&engine, cfg, &dof_indices).unwrap();

    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    lstm.set_weights(&device, &queue, &net.weights()).unwrap();

    let x = vec![0.4 as Real, 0.1, -0.2];

    // CPU rollout: 10 steps of the recurrence with the same input.
    let mut h_cpu = vec![0.0 as Real; net.hid_dim];
    let mut c_cpu = vec![0.0 as Real; net.hid_dim];
    let mut h_after_1: Vec<Real> = Vec::new();
    for step in 1..=10 {
        let (h_n, c_n, _) = cpu_lstm_step(
            &x, &h_cpu, &c_cpu, &net.weights(),
            net.in_dim, net.hid_dim, net.out_dim,
        );
        h_cpu = h_n;
        c_cpu = c_n;
        if step == 1 {
            h_after_1 = h_cpu.clone();
        }
    }
    let h_after_10_cpu = h_cpu.clone();

    // GPU rollout: 10 forwards, no reset between them.
    lstm.set_inputs(&queue, &x).unwrap();
    let mut h_after_1_gpu: Vec<Real> = Vec::new();
    for step in 1..=10 {
        lstm.forward(&engine).unwrap();
        if step == 1 {
            h_after_1_gpu = lstm.download_hidden_state(&device, &queue).unwrap();
        }
    }
    let h_after_10_gpu = lstm.download_hidden_state(&device, &queue).unwrap();

    eprintln!("h after  1 (cpu): {h_after_1:?}");
    eprintln!("h after  1 (gpu): {h_after_1_gpu:?}");
    eprintln!("h after 10 (cpu): {h_after_10_cpu:?}");
    eprintln!("h after 10 (gpu): {h_after_10_gpu:?}");

    let tol = 1.0e-4;
    for i in 0..net.hid_dim {
        assert!((h_after_1_gpu[i] - h_after_1[i]).abs() < tol,
            "h_1[{i}] mismatch: gpu={} cpu={}", h_after_1_gpu[i], h_after_1[i]);
        assert!((h_after_10_gpu[i] - h_after_10_cpu[i]).abs() < tol,
            "h_10[{i}] mismatch: gpu={} cpu={}", h_after_10_gpu[i], h_after_10_cpu[i]);
    }

    // Persistence check: h_10 must differ meaningfully from h_1 — the
    // recurrence has to actually be carrying information forward.
    let mut max_diff = 0.0 as Real;
    for i in 0..net.hid_dim {
        max_diff = max_diff.max((h_after_10_gpu[i] - h_after_1_gpu[i]).abs());
    }
    eprintln!("max |h_10 - h_1| (gpu): {max_diff}");
    assert!(max_diff > 1.0e-3,
        "hidden state didn't evolve across 10 substeps — recurrence \
         likely got reset (max diff {max_diff})");
}

/// Test 3: `reset_hidden_state` zeros both `h` and `c`. We run one
/// step (so the buffers are non-zero), call reset, and verify both
/// host- and device-side mirrors are back to zero.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn lstm_reset_hidden_state_zeros_h_and_c() {
    let Some(engine) = make_engine_or_skip() else { return };

    let net = build_known_net();
    let cfg = ControllerNeuralLstmConfig {
        input_dim: net.in_dim as u32,
        hidden_dim: net.hid_dim as u32,
        output_dim: net.out_dim as u32,
    };
    let dof_indices: Vec<u32> = (0..net.out_dim as u32).collect();
    let mut lstm = ControllerNeuralLstm::new(&engine, cfg, &dof_indices).unwrap();

    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    lstm.set_weights(&device, &queue, &net.weights()).unwrap();

    // Drive the network so h, c become non-zero.
    let x = vec![0.7 as Real, -0.4, 0.5];
    lstm.set_inputs(&queue, &x).unwrap();
    lstm.forward(&engine).unwrap();
    let h_pre = lstm.download_hidden_state(&device, &queue).unwrap();
    let c_pre = lstm.download_cell_state(&device, &queue).unwrap();
    let any_nonzero = |v: &[Real]| v.iter().any(|x| x.abs() > 1.0e-6);
    assert!(any_nonzero(&h_pre), "h was zero before reset — test setup wrong");
    assert!(any_nonzero(&c_pre), "c was zero before reset — test setup wrong");

    // Reset.
    lstm.reset_hidden_state(&queue).unwrap();

    // Confirm both buffers are zero on-device.
    let h_post = lstm.download_hidden_state(&device, &queue).unwrap();
    let c_post = lstm.download_cell_state(&device, &queue).unwrap();
    eprintln!("h after reset: {h_post:?}");
    eprintln!("c after reset: {c_post:?}");
    for (i, v) in h_post.iter().enumerate() {
        assert!(v.abs() < 1.0e-9, "h[{i}] not zero after reset: {v}");
    }
    for (i, v) in c_post.iter().enumerate() {
        assert!(v.abs() < 1.0e-9, "c[{i}] not zero after reset: {v}");
    }
}
