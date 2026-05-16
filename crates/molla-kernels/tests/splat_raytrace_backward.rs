//! Stage 44 v2 — GPU backward kernel parity vs CPU finite-difference
//! oracle for the 3D Gaussian splat raytrace.
//!
//! Each `#[test]` is `#[ignore]` because it needs a real wgpu adapter.
//! Run with `--ignored`:
//!
//! ```text
//! nixVulkan cargo test -p molla-kernels --features gpu \
//!     --test splat_raytrace_backward -- --ignored
//! ```
//!
//! Tests cover each of the five differentiable parameter groups
//! (position, color, opacity, scale, rotation). The setup is one
//! splat placed in front of the camera with a small upstream
//! `dL/dC_out` and the analytic GPU gradient is compared against the
//! central FD gradient on the CPU oracle. Relative-error tolerance is
//! `1e-3` per the v2 contract — well within the f32 noise floor of
//! the GPU kernel but tight enough to catch a wrong chain rule.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_geometry::{GaussianSplat3d, SplatParam, fd_grad_one_splat};
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

/// Build SoA buffers for `splats`, upload, and run the backward
/// kernel against a single-pixel `dL/dC`. Returns the per-splat
/// analytic gradients (downloaded to host).
fn run_backward(
    engine: &Engine,
    splats: &[GaussianSplat3d],
    ray_origin: Vec3,
    ray_dir: Vec3,
    dl_dc: Vec3,
    max_distance: f32,
    min_alpha: f32,
) -> molla_kernels::SplatGradients {
    let device = engine.device().expect("gpu device").clone();
    let queue = engine.queue().expect("gpu queue").clone();

    let pos: Vec<Vec3> = splats.iter().map(|s| s.position).collect();
    let scale: Vec<Vec3> = splats.iter().map(|s| s.scale).collect();
    let rot: Vec<Quat> = splats.iter().map(|s| s.rotation).collect();
    let color: Vec<Vec3> = splats.iter().map(|s| s.color_rgb()).collect();
    let opacity: Vec<Real> = splats.iter().map(|s| s.opacity).collect();

    let mut splat_pos = Buffer::from_vec(pos, BufferUsage::Storage);
    let mut splat_scale = Buffer::from_vec(scale, BufferUsage::Storage);
    let mut splat_rot = Buffer::from_vec(rot, BufferUsage::Storage);
    let mut splat_color = Buffer::from_vec(color, BufferUsage::Storage);
    let mut splat_opacity = Buffer::from_vec(opacity, BufferUsage::Storage);
    splat_pos.ensure_device(&device).unwrap();
    splat_scale.ensure_device(&device).unwrap();
    splat_rot.ensure_device(&device).unwrap();
    splat_color.ensure_device(&device).unwrap();
    splat_opacity.ensure_device(&device).unwrap();
    splat_pos.upload(&queue).unwrap();
    splat_scale.upload(&queue).unwrap();
    splat_rot.upload(&queue).unwrap();
    splat_color.upload(&queue).unwrap();
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
        .backward(
            &device,
            &queue,
            &splat_pos,
            &splat_scale,
            &splat_rot,
            &splat_color,
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

/// Single splat slightly off-axis so every parameter has a non-trivial
/// gradient (a centred ray with the unit-isotropic σ-1 splat zeros out
/// rotation / scale gradients by symmetry).
fn off_axis_splat() -> GaussianSplat3d {
    GaussianSplat3d::new(
        Vec3::new(0.3, -0.2, 0.0),
        Vec3::new(1.2, 0.7, 1.5),
        Quat::from_xyzw(0.1, 0.2, 0.05, 0.97).normalize(),
        Vec3::new(0.4, 0.6, 0.8),
        0.55,
    )
}

fn compare_grads(label: &str, analytic: &[Real], fd: &[Real]) {
    assert_eq!(analytic.len(), fd.len(), "{label}: length mismatch");
    for (i, (a, f)) in analytic.iter().zip(fd.iter()).enumerate() {
        let denom = f.abs().max(1e-6);
        let rel = (a - f).abs() / denom;
        assert!(
            rel < 1e-3,
            "{label}[{i}]: analytic {a:.6e}, fd {f:.6e}, rel = {rel:.3e}",
        );
    }
}

/// Position gradient — perturb x, y, z each by ±h, compare to GPU
/// analytic.
#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn backward_position_gradient_matches_fd() {
    let _gpu = lock_gpu();
    let engine = try_engine().expect("GPU adapter required");
    let s = off_axis_splat();
    let ray_o = Vec3::new(0.0, 0.0, -5.0);
    let ray_d = Vec3::Z;
    let dl_dc = Vec3::new(1.0, 0.5, -0.7);
    let grads = run_backward(&engine, &[s], ray_o, ray_d, dl_dc, 100.0, 1.0 / 255.0);
    let analytic = vec![grads.d_pos[0].x, grads.d_pos[0].y, grads.d_pos[0].z];
    let fd = fd_grad_one_splat(
        &[s],
        0,
        &[SplatParam::PosX, SplatParam::PosY, SplatParam::PosZ],
        ray_o,
        ray_d,
        dl_dc,
        100.0,
        1.0 / 255.0,
        1e-4,
    );
    compare_grads("position", &analytic, &fd);
}

/// Color (rgb) gradient.
#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn backward_color_gradient_matches_fd() {
    let _gpu = lock_gpu();
    let engine = try_engine().expect("GPU adapter required");
    let s = off_axis_splat();
    let ray_o = Vec3::new(0.0, 0.0, -5.0);
    let ray_d = Vec3::Z;
    let dl_dc = Vec3::new(1.0, 0.5, -0.7);
    let grads = run_backward(&engine, &[s], ray_o, ray_d, dl_dc, 100.0, 1.0 / 255.0);
    let analytic = vec![grads.d_color[0].x, grads.d_color[0].y, grads.d_color[0].z];
    let fd = fd_grad_one_splat(
        &[s],
        0,
        &[SplatParam::ColorR, SplatParam::ColorG, SplatParam::ColorB],
        ray_o,
        ray_d,
        dl_dc,
        100.0,
        1.0 / 255.0,
        1e-4,
    );
    compare_grads("color", &analytic, &fd);
}

/// Opacity gradient — scalar.
#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn backward_opacity_gradient_matches_fd() {
    let _gpu = lock_gpu();
    let engine = try_engine().expect("GPU adapter required");
    let s = off_axis_splat();
    let ray_o = Vec3::new(0.0, 0.0, -5.0);
    let ray_d = Vec3::Z;
    let dl_dc = Vec3::new(1.0, 0.5, -0.7);
    let grads = run_backward(&engine, &[s], ray_o, ray_d, dl_dc, 100.0, 1.0 / 255.0);
    let analytic = vec![grads.d_opacity[0]];
    let fd = fd_grad_one_splat(
        &[s],
        0,
        &[SplatParam::Opacity],
        ray_o,
        ray_d,
        dl_dc,
        100.0,
        1.0 / 255.0,
        1e-4,
    );
    compare_grads("opacity", &analytic, &fd);
}

/// Scale (per-axis stddev) gradient.
#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn backward_scale_gradient_matches_fd() {
    let _gpu = lock_gpu();
    let engine = try_engine().expect("GPU adapter required");
    let s = off_axis_splat();
    let ray_o = Vec3::new(0.0, 0.0, -5.0);
    let ray_d = Vec3::Z;
    let dl_dc = Vec3::new(1.0, 0.5, -0.7);
    let grads = run_backward(&engine, &[s], ray_o, ray_d, dl_dc, 100.0, 1.0 / 255.0);
    let analytic = vec![grads.d_scale[0].x, grads.d_scale[0].y, grads.d_scale[0].z];
    let fd = fd_grad_one_splat(
        &[s],
        0,
        &[SplatParam::ScaleX, SplatParam::ScaleY, SplatParam::ScaleZ],
        ray_o,
        ray_d,
        dl_dc,
        100.0,
        1.0 / 255.0,
        1e-4,
    );
    compare_grads("scale", &analytic, &fd);
}

/// Rotation (quaternion) gradient. We compare the 4-component
/// derivative directly — the FD oracle perturbs each (x, y, z, w)
/// independently without renormalising, so it sees exactly the same
/// non-projected gradient the kernel computes. Callers who need the
/// tangent of the unit-sphere project after the fact.
#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn backward_rotation_gradient_matches_fd() {
    let _gpu = lock_gpu();
    let engine = try_engine().expect("GPU adapter required");
    let s = off_axis_splat();
    let ray_o = Vec3::new(0.0, 0.0, -5.0);
    let ray_d = Vec3::Z;
    let dl_dc = Vec3::new(1.0, 0.5, -0.7);
    let grads = run_backward(&engine, &[s], ray_o, ray_d, dl_dc, 100.0, 1.0 / 255.0);
    let analytic = vec![
        grads.d_rotation[0].x,
        grads.d_rotation[0].y,
        grads.d_rotation[0].z,
        grads.d_rotation[0].w,
    ];
    let fd = fd_grad_one_splat(
        &[s],
        0,
        &[
            SplatParam::RotX,
            SplatParam::RotY,
            SplatParam::RotZ,
            SplatParam::RotW,
        ],
        ray_o,
        ray_d,
        dl_dc,
        100.0,
        1.0 / 255.0,
        1e-4,
    );

    // Quaternion gradient parity is harder than the others: the
    // analytic kernel computes ∂L/∂q on the **unconstrained** R⁴
    // quaternion (treating q as free), while the FD oracle does the
    // same thing without renormalising. The two should match up to
    // f32 round-off. We use a slightly relaxed tolerance (1e-3 rel /
    // 5e-4 abs) because the rotation gradient flows through both
    // (oR, dR) and is the most cancellation-prone of the five.
    assert_eq!(analytic.len(), fd.len(), "rotation: length mismatch");
    for (i, (a, f)) in analytic.iter().zip(fd.iter()).enumerate() {
        let denom = f.abs().max(5e-4);
        let rel = (a - f).abs() / denom;
        assert!(
            rel < 1e-3,
            "rotation[{i}]: analytic {a:.6e}, fd {f:.6e}, rel = {rel:.3e}",
        );
    }
}
