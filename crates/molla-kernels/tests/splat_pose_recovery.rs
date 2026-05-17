//! Stage 44 v2 — end-to-end camera-pose recovery demo.
//!
//! Render a target image of 1k random Gaussian splats at a known
//! camera pose; perturb the camera by 1° + 1 cm; run Adam (lr 1e-3)
//! on the (translation, rotation) pair using the rendered-vs-target
//! L2 pixel loss; assert recovery within 0.1° + 1 mm in ≤ 500 iters.
//!
//! The camera pose enters through a fixed-camera / moving-splat
//! parameterisation: every iteration we transform each splat
//! (position + rotation) into camera coordinates using the current
//! `(t_c, q_c)` and dispatch the GPU forward/backward over **fixed
//! canonical rays** (origin = 0, direction = +Z, plus jitter for the
//! pixel grid). The kernel-returned per-splat `d_pos` is then
//! chained back to the camera transform via the standard rigid-body
//! Jacobian:
//!
//! ```text
//!   dL/dt_world  =  - R_c · Σ_i  d_pos_i^{cam}
//!   dL/dω_cam    =  - Σ_i  pos_cam_i × d_pos_i^{cam}
//! ```
//!
//! where `ω_cam` is a small-angle deviation in the camera's local
//! frame and updates compose as `q_c ← q_c · exp(δω)`. (Splat-
//! rotation gradients are computed by the kernel but not chained
//! back here — the splats in this demo are nearly isotropic, so
//! their rotation gradient barely couples to camera rotation. The
//! position-gradient → rigid-body Jacobian is the dominant signal.)
//!
//! `#[ignore]` because it needs a GPU adapter; run with
//! `nixVulkan cargo test -p molla-kernels --features gpu --release \
//!     --test splat_pose_recovery -- --ignored`.

#![cfg(feature = "gpu")]
#![allow(non_snake_case)] // math notation (dL_dC, dL_dt, ...).

use std::sync::Mutex;

use molla_compute::Engine;
use molla_geometry::GaussianSplat3d;
use molla_kernels::SplatRaytraceKernel;
use molla_math::{Quat, Real, Vec3};
use molla_optim::{AdamConfig, AdamState};
use molla_storage::{Buffer, BufferUsage};

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn lock_gpu() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

/// Tiny deterministic LCG so the test is reproducible without an
/// `rand` dependency. Returns a value in `[0, 1)`.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> Real {
        // Numerical Recipes 64-bit LCG parameters.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Top 53 bits to f64.
        ((self.0 >> 11) as Real) / ((1u64 << 53) as Real)
    }
    fn uniform(&mut self, lo: Real, hi: Real) -> Real {
        lo + (hi - lo) * self.next()
    }
}

/// Build N splats sprinkled inside a `±1` cube in front of the
/// origin, with random colours, opacities, and modest anisotropy.
fn make_scene(n: usize, seed: u64) -> Vec<GaussianSplat3d> {
    let mut rng = Lcg::new(seed);
    let mut splats = Vec::with_capacity(n);
    for _ in 0..n {
        let pos = Vec3::new(
            rng.uniform(-0.8, 0.8),
            rng.uniform(-0.6, 0.6),
            rng.uniform(0.5, 3.0),
        );
        let s = rng.uniform(0.03, 0.08);
        let scale = Vec3::new(s, s * rng.uniform(0.7, 1.3), s * rng.uniform(0.7, 1.3));
        // Random unit quaternion via 4 gaussian-ish draws + normalize.
        let q_raw = Quat::from_xyzw(
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
        );
        let q = q_raw.normalize();
        let color = Vec3::new(rng.next(), rng.next(), rng.next());
        let opacity = rng.uniform(0.4, 0.9);
        splats.push(GaussianSplat3d::new(pos, scale, q, color, opacity));
    }
    splats
}

/// Build a regular pixel grid of size `WxH` looking down `+Z` with
/// the standard pinhole approximation. Returns (origins, dirs) in
/// canonical camera coords.
fn make_rays(w: usize, h: usize, fov_y: Real) -> (Vec<Vec3>, Vec<Vec3>) {
    let aspect = w as Real / h as Real;
    let tan_half = (fov_y * 0.5).tan();
    let mut origins = Vec::with_capacity(w * h);
    let mut dirs = Vec::with_capacity(w * h);
    for j in 0..h {
        for i in 0..w {
            // Normalised pixel coords in [-1, +1].
            let u = (i as Real + 0.5) / w as Real * 2.0 - 1.0;
            let v = (j as Real + 0.5) / h as Real * 2.0 - 1.0;
            let x = u * tan_half * aspect;
            let y = v * tan_half;
            let dir = Vec3::new(x, y, 1.0); // not normalised — kernel uses dot/|d|² so length cancels
            origins.push(Vec3::new(0.0, 0.0, 0.0));
            dirs.push(dir);
        }
    }
    (origins, dirs)
}

/// Same quat-rotate formula the kernel + CPU oracle use (BAC-CAB).
/// Equivalent to `q * v` for unit q; we keep it explicit because the
/// pose-recovery loop may push q transiently off the unit sphere
/// between renormalisations.
fn quat_rotate_bac(q: Quat, v: Vec3) -> Vec3 {
    let qv = Vec3::new(q.x, q.y, q.z);
    let t = qv.cross(v) * 2.0;
    v + t * q.w + qv.cross(t)
}

fn quat_conj(q: Quat) -> Quat {
    Quat::from_xyzw(-q.x, -q.y, -q.z, q.w)
}

/// Transform a world-frame splat into the camera's frame at pose
/// `(t_c, q_c)`. The camera convention is `pos_cam = R_c^T (pos -
/// t_c)`, `rot_cam = q_c^{-1} * rot`. Position and orientation move
/// with the rigid body; per-axis scale + colour + opacity are
/// camera-invariant.
fn transform_splat_to_camera(s: &GaussianSplat3d, t_c: Vec3, q_c: Quat) -> GaussianSplat3d {
    let q_inv = quat_conj(q_c);
    let pos_cam = quat_rotate_bac(q_inv, s.position - t_c);
    // q_inv * rot via standard quaternion product.
    let rot_cam = q_inv * s.rotation;
    GaussianSplat3d::new(
        pos_cam,
        s.scale,
        rot_cam.normalize(),
        s.color_rgb(),
        s.opacity,
    )
}

/// Wrap a `Vec<GaussianSplat3d>` into the SoA storage buffers the
/// GPU kernel consumes. Uploads them and leaves them device-resident.
fn upload_splat_soa(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    splats: &[GaussianSplat3d],
) -> (
    Buffer<Vec3>,
    Buffer<Vec3>,
    Buffer<Quat>,
    Buffer<Vec3>,
    Buffer<Real>,
) {
    let pos: Vec<Vec3> = splats.iter().map(|s| s.position).collect();
    let scale: Vec<Vec3> = splats.iter().map(|s| s.scale).collect();
    let rot: Vec<Quat> = splats.iter().map(|s| s.rotation).collect();
    let color: Vec<Vec3> = splats.iter().map(|s| s.color_rgb()).collect();
    let opacity: Vec<Real> = splats.iter().map(|s| s.opacity).collect();
    let mut bp = Buffer::from_vec(pos, BufferUsage::Storage);
    let mut bs = Buffer::from_vec(scale, BufferUsage::Storage);
    let mut br = Buffer::from_vec(rot, BufferUsage::Storage);
    let mut bc = Buffer::from_vec(color, BufferUsage::Storage);
    let mut bo = Buffer::from_vec(opacity, BufferUsage::Storage);
    bp.ensure_device(device).unwrap();
    bs.ensure_device(device).unwrap();
    br.ensure_device(device).unwrap();
    bc.ensure_device(device).unwrap();
    bo.ensure_device(device).unwrap();
    bp.upload(queue).unwrap();
    bs.upload(queue).unwrap();
    br.upload(queue).unwrap();
    bc.upload(queue).unwrap();
    bo.upload(queue).unwrap();
    (bp, bs, br, bc, bo)
}

/// Forward render — returns the per-pixel rgba buffer (downloaded).
#[allow(clippy::too_many_arguments)]
fn render_image(
    kernel: &SplatRaytraceKernel,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    splats: &[GaussianSplat3d],
    rays_o: &[Vec3],
    rays_d: &[Vec3],
    max_d: f32,
    min_alpha: f32,
) -> Vec<Vec3> {
    let (bp, bs, br, bc, bo) = upload_splat_soa(device, queue, splats);
    let mut ro = Buffer::from_vec(rays_o.to_vec(), BufferUsage::Storage);
    let mut rd = Buffer::from_vec(rays_d.to_vec(), BufferUsage::Storage);
    ro.ensure_device(device).unwrap();
    rd.ensure_device(device).unwrap();
    ro.upload(queue).unwrap();
    rd.upload(queue).unwrap();
    let n = rays_o.len();
    let mut out_rgba = Buffer::from_vec(vec![Vec3::ZERO; n], BufferUsage::StorageReadWrite);
    let mut out_depth = Buffer::from_vec(vec![-1.0_f64; n], BufferUsage::StorageReadWrite);
    out_rgba.ensure_device(device).unwrap();
    out_depth.ensure_device(device).unwrap();
    out_rgba.upload(queue).unwrap();
    out_depth.upload(queue).unwrap();
    kernel
        .dispatch(
            device,
            queue,
            &bp,
            &bs,
            &br,
            &bc,
            &bo,
            &ro,
            &rd,
            &mut out_rgba,
            &mut out_depth,
            n as u32,
            max_d,
            min_alpha,
        )
        .unwrap();
    out_rgba.download(device, queue).unwrap();
    out_rgba.host().unwrap().to_vec()
}

#[allow(clippy::too_many_arguments)]
fn backward_gradients(
    kernel: &SplatRaytraceKernel,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    splats: &[GaussianSplat3d],
    rays_o: &[Vec3],
    rays_d: &[Vec3],
    dL_dC: &[Vec3],
    max_d: f32,
    min_alpha: f32,
) -> molla_kernels::SplatGradients {
    let (bp, bs, br, bc, bo) = upload_splat_soa(device, queue, splats);
    let mut ro = Buffer::from_vec(rays_o.to_vec(), BufferUsage::Storage);
    let mut rd = Buffer::from_vec(rays_d.to_vec(), BufferUsage::Storage);
    let mut dL = Buffer::from_vec(dL_dC.to_vec(), BufferUsage::Storage);
    ro.ensure_device(device).unwrap();
    rd.ensure_device(device).unwrap();
    dL.ensure_device(device).unwrap();
    ro.upload(queue).unwrap();
    rd.upload(queue).unwrap();
    dL.upload(queue).unwrap();
    kernel
        .backward(
            device,
            queue,
            &bp,
            &bs,
            &br,
            &bc,
            &bo,
            &ro,
            &rd,
            &dL,
            rays_o.len() as u32,
            max_d,
            min_alpha,
        )
        .unwrap()
}

/// Rotate a camera-frame vector into world coords by the current
/// camera quaternion. Used to chain a sum of camera-frame splat
/// position gradients into the world-frame translation gradient.
fn rotate_camera_to_world(q_c: Quat, v_cam: Vec3) -> Vec3 {
    quat_rotate_bac(q_c, v_cam)
}

/// Apply a tiny right-handed rotation (axis × angle, both in `omega`)
/// to `q`. For `omega = (0, 0, 0)` returns `q` unchanged.
fn apply_omega(q: Quat, omega: Vec3) -> Quat {
    let angle = omega.length();
    if angle < 1e-12 {
        return q;
    }
    let axis = omega / angle;
    let half = 0.5 * angle;
    let s = half.sin();
    let c = half.cos();
    let dq = Quat::from_xyzw(axis.x * s, axis.y * s, axis.z * s, c);
    (q * dq).normalize()
}

/// Pose recovery — render a 1k-splat target image, perturb the
/// camera by 1° + 1 cm, run Adam (lr 1e-3) for ≤ 500 iters and
/// assert convergence within 0.1° + 1 mm.
#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn splat_pose_recovery_within_tolerance() {
    let _gpu = lock_gpu();
    let engine = try_engine().expect("GPU adapter required");
    let device = engine.device().expect("gpu device").clone();
    let queue = engine.queue().expect("gpu queue").clone();
    let kernel = SplatRaytraceKernel::new(&device);

    // ── Scene ─────────────────────────────────────────────────
    let splats_world = make_scene(1024, 0xC0FFEE_BEEF_DEAD);

    // 32x24 = 768 pixel grid; small but enough to constrain a
    // 6-DoF pose. Smaller grids = faster GPU dispatch.
    let (w, h) = (32usize, 24usize);
    let (rays_o, rays_d) = make_rays(w, h, 60_f64.to_radians());

    // ── Target image at the **truth** camera pose ─────────────
    // Truth: camera at origin, looking +Z (identity rotation).
    let t_truth = Vec3::ZERO;
    let q_truth = Quat::IDENTITY;
    let splats_truth_cam: Vec<GaussianSplat3d> = splats_world
        .iter()
        .map(|s| transform_splat_to_camera(s, t_truth, q_truth))
        .collect();
    let target = render_image(
        &kernel,
        &device,
        &queue,
        &splats_truth_cam,
        &rays_o,
        &rays_d,
        100.0,
        1.0 / 255.0,
    );

    // ── Initial guess — 1 cm + 1° offset. ─────────────────────
    // Translation 1 cm along +X. Rotation 1° about (1,1,1)/√3.
    let mut t_c = Vec3::new(0.01, 0.0, 0.0);
    let axis = Vec3::new(1.0, 1.0, 1.0).normalize();
    let angle = 1_f64.to_radians();
    let init_omega = axis * angle;
    let mut q_c = apply_omega(Quat::IDENTITY, init_omega);

    // ── Optimizer: 6-vector (Δt_world.xyz, ω_cam.xyz). ────────
    // We treat the rotation update as a tangent vector in camera-
    // local axis-angle space and compose it into q_c each step.
    let mut params: Vec<Real> = vec![0.0; 6];
    let mut adam = AdamState::new(6);
    let cfg = AdamConfig {
        lr: 1e-3,
        ..AdamConfig::default()
    };

    let max_iters = 500;
    let mut final_t_err = Real::INFINITY;
    let mut final_r_err = Real::INFINITY;

    for it in 0..max_iters {
        // (1) Transform splats into camera frame.
        let splats_cam: Vec<GaussianSplat3d> = splats_world
            .iter()
            .map(|s| transform_splat_to_camera(s, t_c, q_c))
            .collect();
        // (2) Forward render.
        let img = render_image(
            &kernel,
            &device,
            &queue,
            &splats_cam,
            &rays_o,
            &rays_d,
            100.0,
            1.0 / 255.0,
        );
        // (3) L2 pixel residual (no factor of ½).
        let mut loss: Real = 0.0;
        let mut dL_dC: Vec<Vec3> = Vec::with_capacity(img.len());
        for (cur, tgt) in img.iter().zip(target.iter()) {
            let r = *cur - *tgt;
            loss += r.length_squared();
            dL_dC.push(r * 2.0); // gradient of `Σ r²`
        }
        // (4) Backward — per-splat gradients in **camera frame**.
        let grads = backward_gradients(
            &kernel,
            &device,
            &queue,
            &splats_cam,
            &rays_o,
            &rays_d,
            &dL_dC,
            100.0,
            1.0 / 255.0,
        );

        // (5) Chain → camera Jacobian.
        // dL/dt_world = - R_c · Σ_i d_pos_i^{cam}
        // dL/dω_cam   = - Σ_i  pos_cam_i × d_pos_i^{cam}
        let mut sum_dpos = Vec3::ZERO;
        let mut sum_cross = Vec3::ZERO;
        for (s_cam, dp) in splats_cam.iter().zip(grads.d_pos.iter()) {
            sum_dpos += *dp;
            sum_cross += s_cam.position.cross(*dp);
        }
        let dL_dt = -rotate_camera_to_world(q_c, sum_dpos);
        let dL_domega = -sum_cross;

        // (6) Adam step. The 6-vector `params` is the cumulative
        // (Δt, Δω) Adam thinks is best; we materialise it onto
        // (t_c, q_c) each step. To make this work with Adam's
        // moment estimates we re-baseline: after each step, fold
        // the params *back* into (t_c, q_c) and zero params (this
        // is equivalent to running Adam on the increment and
        // applying it immediately — moments still adapt to the
        // residual gradient direction).
        let grad = [
            dL_dt.x,
            dL_dt.y,
            dL_dt.z,
            dL_domega.x,
            dL_domega.y,
            dL_domega.z,
        ];
        adam.step(&mut params, &grad, &cfg);
        let delta_t = Vec3::new(params[0], params[1], params[2]);
        let delta_omega = Vec3::new(params[3], params[4], params[5]);
        t_c += delta_t;
        q_c = apply_omega(q_c, delta_omega);
        for p in &mut params {
            *p = 0.0;
        }

        // (7) Measure pose error vs truth.
        let t_err = (t_c - t_truth).length();
        let r_err = {
            // angle between q_c and identity (truth) in radians.
            let dq = q_truth * quat_conj(q_c);
            let dq = dq.normalize();
            let cosang = dq.w.abs().clamp(-1.0, 1.0);
            2.0 * cosang.acos()
        };
        final_t_err = t_err;
        final_r_err = r_err;
        if it % 50 == 0 || it == max_iters - 1 {
            eprintln!(
                "iter {it:4}: loss = {loss:.4e}, |t_err| = {tmm:.4} mm, |r_err| = {rdeg:.4}°",
                tmm = t_err * 1000.0,
                rdeg = r_err.to_degrees(),
            );
        }
        // Early-exit if we've converged.
        if t_err < 1e-3 && r_err < 0.1_f64.to_radians() && it > 50 {
            break;
        }
    }

    let t_err_mm = final_t_err * 1000.0;
    let r_err_deg = final_r_err.to_degrees();
    eprintln!("final: |t_err| = {t_err_mm:.4} mm, |r_err| = {r_err_deg:.4}°");
    assert!(
        t_err_mm < 1.0,
        "translation error {t_err_mm:.4} mm exceeds 1 mm budget"
    );
    assert!(
        r_err_deg < 0.1,
        "rotation error {r_err_deg:.4}° exceeds 0.1° budget"
    );
}
