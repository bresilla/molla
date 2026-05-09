//! Stage 23 demo — trajectory optimisation on a 2-link planar arm
//! via gradient descent on an open-loop torque sequence.
//!
//! The arm dynamics + closed-form adjoint live in
//! [`molla::autograd::DiffArm`] so the example, the lib tests, and
//! the viewer's `Scene::DiffArm` can all share the same primitive.
//!
//! Run:
//!     cargo run -p molla --example diff_arm_reach --features gpu
//!
//! Expected output: the residual converges below 5 cm within ~20
//! iterations of GD with the default arm and `lr = 5.0`.

use molla::autograd::DiffArm;
use molla::math::Real;

const LR: Real = 5.0;
const ITERS: usize = 100;

fn main() {
    let arm = DiffArm::default();
    println!("molla::autograd diff_arm_reach demo");
    println!(
        "  L1 = {} m, L2 = {} m, inertia = {}, damping = {}",
        arm.l1, arm.l2, arm.inertia, arm.damping,
    );
    println!(
        "  horizon = {} substeps × dt = {} s ({} s total)",
        arm.horizon, arm.dt, arm.horizon as Real * arm.dt,
    );
    println!("  optimisation: {ITERS} iters of GD with lr = {LR}");
    println!();

    let target: [Real; 2] = [0.45, -0.25];
    println!("target EE = ({}, {})", target[0], target[1]);
    println!();

    let mut tau = vec![[0.0_f64; 2]; arm.horizon];
    let q0 = [0.0; 2];
    let qd0 = [0.0; 2];

    // Verbose loop: re-run the inner step 10 iters at a time so we
    // can print intermediate residuals. The DiffArm::gradient_descent
    // helper is the one-call version.
    let mut last = Real::INFINITY;
    let mut converged_at: Option<usize> = None;
    for it in 0..ITERS {
        let trace = arm.rollout(q0, qd0, &tau);
        let ee = arm.forward_kinematics(trace[trace.len() - 1].0);
        let dx = ee[0] - target[0];
        let dy = ee[1] - target[1];
        let residual = (dx * dx + dy * dy).sqrt();
        if it % 10 == 0 || it == ITERS - 1 {
            println!(
                "iter {it:>3}: ee = ({:>+.4}, {:>+.4})  residual = {:.4} m",
                ee[0], ee[1], residual,
            );
        }
        if converged_at.is_none() && residual < 0.05 {
            converged_at = Some(it);
        }
        last = residual;
        let g_tau = arm.reverse(&trace, [2.0 * dx, 2.0 * dy]);
        for i in 0..arm.horizon {
            tau[i][0] -= LR * g_tau[i][0];
            tau[i][1] -= LR * g_tau[i][1];
        }
    }

    println!();
    println!("final residual: {last:.4} m (sanity tolerance: 0.05 m)");
    if let Some(it) = converged_at {
        println!("converged within 5 cm of target at iter {it}");
    } else {
        println!(
            "did NOT converge within 5 cm in {ITERS} iters — try a different LR or HORIZON",
        );
    }
}
