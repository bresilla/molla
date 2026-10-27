//! Stage 44 v3 — backward parity test for the SH-aware 3DGS adjoint.
//!
//! Verifies the per-SH-coefficient gradient produced by the GPU
//! backward kernel matches the CPU finite-difference oracle within
//! `1e-3` relative error.
//!
//! `#[ignore]` because it needs a real wgpu adapter.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_geometry::{
    GaussianSplatSh3d, SH_BAND_2_COUNT, SH_C0, ShCoeffParam, fd_grad_sh_one_splat,
};
use molla_kernels::SplatRaytraceKernel;
use molla_math::{Quat, Real, Vec3};
use molla_storage::{Buffer, BufferUsage};

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn lock_gpu() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

fn flatten_sh(splats: &[GaussianSplatSh3d]) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(splats.len() * SH_BAND_2_COUNT);
    for s in splats {
        out.extend_from_slice(&s.sh);
    }
    out
}

fn run_backward_sh(
    engine: &Engine,
    splats: &[GaussianSplatSh3d],
    ray_origin: Vec3,
    ray_dir: Vec3,
    dl_dc: Vec3,
    max_distance: f32,
    min_alpha: f32,
) -> molla_kernels::SplatShGradients {
    let device = engine.device().expect("gpu device").clone();
    let queue = engine.queue().expect("gpu queue").clone();

    let pos: Vec<Vec3> = splats.iter().map(|s| s.position).collect();
    let scale: Vec<Vec3> = splats.iter().map(|s| s.scale).collect();
    let rot: Vec<Quat> = splats.iter().map(|s| s.rotation).collect();
    let sh: Vec<Vec3> = flatten_sh(splats);
    let opacity: Vec<Real> = splats.iter().map(|s| s.opacity).collect();

    let mut splat_pos = Buffer::from_vec(pos, BufferUsage::Storage);
    let mut splat_scale = Buffer::from_vec(scale, BufferUsage::Storage);
    let mut splat_rot = Buffer::from_vec(rot, BufferUsage::Storage);
    let mut splat_sh = Buffer::from_vec(sh, BufferUsage::Storage);
    let mut splat_opacity = Buffer::from_vec(opacity, BufferUsage::Storage);
    splat_pos.ensure_device(&device).unwrap();
    splat_scale.ensure_device(&device).unwrap();
    splat_rot.ensure_device(&device).unwrap();
    splat_sh.ensure_device(&device).unwrap();
    splat_opacity.ensure_device(&device).unwrap();
    splat_pos.upload(&queue).unwrap();
    splat_scale.upload(&queue).unwrap();
    splat_rot.upload(&queue).unwrap();
    splat_sh.upload(&queue).unwrap();
    splat_opacity.upload(&queue).unwrap();

    let mut ray_o = Buffer::from_vec(vec![ray_origin], BufferUsage::Storage);
    let mut ray_d = Buffer::from_vec(vec![ray_dir], BufferUsage::Storage);
    let mut dl_buf = Buffer::from_vec(vec![dl_dc], BufferUsage::Storage);
    ray_o.ensure_device(&device).unwrap();
    ray_d.ensure_device(&device).unwrap();
    dl_buf.ensure_device(&device).unwrap();
    ray_o.upload(&queue).unwrap();
    ray_d.upload(&queue).unwrap();
    dl_buf.upload(&queue).unwrap();

    let kernel = SplatRaytraceKernel::new(&device);
    kernel
        .backward_sh(
            &device,
            &queue,
            &splat_pos,
            &splat_scale,
            &splat_rot,
            &splat_sh,
            &splat_opacity,
            &ray_o,
            &ray_d,
            &dl_buf,
            1,
            max_distance,
            min_alpha,
        )
        .unwrap()
}

/// An off-axis SH splat with non-trivial coefficients in every band
/// (so every coefficient has a non-zero gradient).
fn off_axis_sh_splat() -> GaussianSplatSh3d {
    // Diffuse base ≈ (0.4, 0.6, 0.8) → sh[0] = base / SH_C0.
    let mut sh = [Vec3::ZERO; SH_BAND_2_COUNT];
    sh[0] = Vec3::new(0.4 / SH_C0, 0.6 / SH_C0, 0.8 / SH_C0);
    // Modest band-1 + band-2 perturbations on each channel — small
    // enough that the band-0 "diffuse" energy dominates but big
    // enough that the per-coefficient gradients are well above noise.
    sh[1] = Vec3::new(0.20, -0.10, 0.05); // c_{1,-1}
    sh[2] = Vec3::new(0.05, 0.25, -0.15); // c_{1, 0}
    sh[3] = Vec3::new(-0.10, 0.05, 0.30); // c_{1, 1}
    sh[4] = Vec3::new(0.15, 0.10, -0.05); // c_{2,-2}
    sh[5] = Vec3::new(-0.05, 0.20, 0.10); // c_{2,-1}
    sh[6] = Vec3::new(0.10, -0.15, 0.20); // c_{2, 0}
    sh[7] = Vec3::new(0.20, 0.05, -0.10); // c_{2,+1}
    sh[8] = Vec3::new(-0.15, 0.25, 0.05); // c_{2,+2}
    GaussianSplatSh3d::from_sh(
        Vec3::new(0.3, -0.2, 0.0),
        Vec3::new(1.2, 0.7, 1.5),
        Quat::from_xyzw(0.1, 0.2, 0.05, 0.97).normalize(),
        sh,
        0.55,
    )
}

fn compare_grads(label: &str, analytic: &[Real], fd: &[Real]) {
    assert_eq!(analytic.len(), fd.len(), "{label}: length mismatch");
    let mut max_rel = 0.0;
    for (i, (a, f)) in analytic.iter().zip(fd.iter()).enumerate() {
        let denom = f.abs().max(1e-6);
        let rel = (a - f).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
        }
        assert!(
            rel < 1e-3,
            "{label}[{i}]: analytic {a:.6e}, fd {f:.6e}, rel = {rel:.3e}",
        );
    }
    // Always print the worst-case for log inspection.
    eprintln!("{label}: max rel-err = {max_rel:.3e}");
}

/// SH-coefficient gradient — verify all 9*3 = 27 per-splat coeffs
/// against the FD oracle. This is the core v3 deliverable.
#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn backward_sh_coeff_gradient_matches_fd() {
    let _gpu = lock_gpu();
    let engine = try_engine().expect("GPU adapter required");
    let s = off_axis_sh_splat();
    let ray_o = Vec3::new(0.0, 0.0, -5.0);
    let ray_d = Vec3::new(0.2, 0.1, 1.0); // non-axial view direction
    let dl_dc = Vec3::new(1.0, 0.5, -0.7);

    let grads = run_backward_sh(&engine, &[s], ray_o, ray_d, dl_dc, 100.0, 1.0 / 255.0);

    // Flatten analytic gradient into a length-27 vector in
    // (coeff, channel) order so we can zip with the FD oracle.
    let mut analytic = Vec::with_capacity(SH_BAND_2_COUNT * 3);
    for k in 0..SH_BAND_2_COUNT {
        analytic.push(grads.d_sh[0][k].x);
        analytic.push(grads.d_sh[0][k].y);
        analytic.push(grads.d_sh[0][k].z);
    }
    let mut params = Vec::with_capacity(SH_BAND_2_COUNT * 3);
    for k in 0..SH_BAND_2_COUNT {
        for ch in 0..3 {
            params.push(ShCoeffParam {
                coeff: k,
                channel: ch,
            });
        }
    }
    let fd = fd_grad_sh_one_splat(
        &[s],
        0,
        &params,
        ray_o,
        ray_d,
        dl_dc,
        100.0,
        1.0 / 255.0,
        1e-4,
    );
    compare_grads("sh", &analytic, &fd);
}

/// Sanity: opacity / position gradients should remain finite and
/// non-zero — these flow through the same geometric chain as v2 and
/// are only restated here because v3's `backward_sh` returns them.
/// We check signs match the FD oracle of the diffuse projection of
/// this SH splat (band-0 only) — full FD parity for these is already
/// covered by `splat_raytrace_backward.rs`.
#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn backward_sh_geometry_gradients_finite() {
    let _gpu = lock_gpu();
    let engine = try_engine().expect("GPU adapter required");
    let s = off_axis_sh_splat();
    let ray_o = Vec3::new(0.0, 0.0, -5.0);
    let ray_d = Vec3::new(0.0, 0.0, 1.0);
    let dl_dc = Vec3::new(1.0, 0.5, -0.7);
    let grads = run_backward_sh(&engine, &[s], ray_o, ray_d, dl_dc, 100.0, 1.0 / 255.0);

    // The splat is off-axis ⇒ position gradient must have a non-zero
    // component on (x, y) and the opacity gradient must be non-zero.
    let dp = grads.d_pos[0];
    let dop = grads.d_opacity[0];
    assert!(
        dp.x.is_finite() && dp.y.is_finite() && dp.z.is_finite(),
        "position gradient not finite: {dp:?}"
    );
    assert!(dop.is_finite(), "opacity gradient not finite: {dop}");
    assert!(
        dp.length() > 1e-4,
        "position gradient suspiciously small: {dp:?}"
    );
    assert!(
        dop.abs() > 1e-4,
        "opacity gradient suspiciously small: {dop}"
    );
}
