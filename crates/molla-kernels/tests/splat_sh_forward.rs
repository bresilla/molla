//! Stage 44 v3 — forward-only GPU test for view-dependent (spherical-
//! harmonics) 3DGS raytrace.
//!
//! Tests:
//! 1. **Band-0 / diffuse parity**: a single splat with `sh[0] = c /
//!    SH_C0` (and zero higher bands) reproduces the v1 diffuse-RGB
//!    forward composite within f32 noise.
//! 2. **Band-1 view dependence**: a splat with a non-zero band-1
//!    coefficient produces a colour that varies with viewing
//!    direction; we render through two different ray directions
//!    (both passing through the splat centre) and verify the rendered
//!    colours differ by the expected SH-basis gradient.
//!
//! Each `#[test]` is `#[ignore]` because it needs a real wgpu adapter.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_geometry::{
    GaussianSplat3d, GaussianSplatSh3d, SH_BAND_2_COUNT, SH_C1, composite_forward_cpu,
    composite_sh_forward_cpu, eval_sh,
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

/// Flatten SH coeffs of `splats` into a single `Vec<Vec3>` of length
/// `splat_count * SH_BAND_2_COUNT` in band-major order (matches
/// `splat_sh[i*9 + k]` on the GPU).
fn flatten_sh(splats: &[GaussianSplatSh3d]) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(splats.len() * SH_BAND_2_COUNT);
    for s in splats {
        out.extend_from_slice(&s.sh);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_forward_sh(
    engine: &Engine,
    splats: &[GaussianSplatSh3d],
    rays_origin: &[Vec3],
    rays_dir: &[Vec3],
    max_distance: f32,
    min_alpha: f32,
) -> (Vec<Vec3>, Vec<Real>) {
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

    let mut ray_origin = Buffer::from_vec(rays_origin.to_vec(), BufferUsage::Storage);
    let mut ray_direction = Buffer::from_vec(rays_dir.to_vec(), BufferUsage::Storage);
    ray_origin.ensure_device(&device).unwrap();
    ray_direction.ensure_device(&device).unwrap();
    ray_origin.upload(&queue).unwrap();
    ray_direction.upload(&queue).unwrap();

    let n = rays_origin.len();
    let mut out_rgba = Buffer::from_vec(vec![Vec3::ZERO; n], BufferUsage::StorageReadWrite);
    let mut out_depth = Buffer::from_vec(vec![-1.0_f64; n], BufferUsage::StorageReadWrite);
    out_rgba.ensure_device(&device).unwrap();
    out_depth.ensure_device(&device).unwrap();
    out_rgba.upload(&queue).unwrap();
    out_depth.upload(&queue).unwrap();

    let kernel = SplatRaytraceKernel::new(&device);
    kernel
        .dispatch_sh(
            &device,
            &queue,
            &splat_pos,
            &splat_scale,
            &splat_rot,
            &splat_sh,
            &splat_opacity,
            &ray_origin,
            &ray_direction,
            &mut out_rgba,
            &mut out_depth,
            n as u32,
            max_distance,
            min_alpha,
        )
        .unwrap();

    out_rgba.download(&device, &queue).unwrap();
    out_depth.download(&device, &queue).unwrap();

    (
        out_rgba.host().unwrap().to_vec(),
        out_depth.host().unwrap().to_vec(),
    )
}

/// 1. Band-0 ("diffuse") sanity. SH coeff `sh[0] = c / SH_C0` →
/// rendered colour ≈ `c · opacity` (ρ_max = 1 for the centred ray).
#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn sh_band0_matches_diffuse_forward() {
    let _gpu = lock_gpu();
    let engine = try_engine().expect("GPU adapter required");

    // Build a diffuse splat and the SH-equivalent splat.
    let rgb = Vec3::new(0.7, 0.3, 0.1);
    let diffuse = GaussianSplat3d::new(
        Vec3::new(0.2, -0.1, 0.0),
        Vec3::new(1.0, 1.0, 1.0),
        Quat::IDENTITY,
        rgb,
        0.6,
    );
    let sh_splat =
        GaussianSplatSh3d::diffuse(diffuse.position, diffuse.scale, diffuse.rotation, rgb, 0.6);

    let origin = Vec3::new(0.0, 0.0, -5.0);
    let dir = Vec3::Z;

    let (rgba_sh, _depth) =
        run_forward_sh(&engine, &[sh_splat], &[origin], &[dir], 100.0, 1.0 / 255.0);

    // Compare against the v1 CPU oracle (well-tested).
    let (rgb_cpu, _t_cpu) = composite_forward_cpu(origin, dir, &[diffuse], 100.0, 1.0 / 255.0);
    let got = rgba_sh[0];
    let drgb = (got - rgb_cpu).length();
    assert!(
        drgb < 5e-3,
        "v1 diffuse {rgb_cpu:?} vs SH band-0 GPU {got:?} (Δ = {drgb})"
    );

    // Also compare to the SH CPU oracle.
    let (rgb_sh_cpu, _) = composite_sh_forward_cpu(origin, dir, &[sh_splat], 100.0, 1.0 / 255.0);
    let dsh = (got - rgb_sh_cpu).length();
    assert!(
        dsh < 5e-3,
        "SH CPU {rgb_sh_cpu:?} vs SH GPU {got:?} (Δ = {dsh})"
    );
}

/// 2. Band-1 view dependence. A pure band-1 splat returns a colour
/// proportional to the view direction's `(x, y, z)`; firing rays
/// along +Z vs +X (both passing through the splat) gives different
/// rendered colours.
#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn sh_band1_view_dependent_forward() {
    let _gpu = lock_gpu();
    let engine = try_engine().expect("GPU adapter required");

    // Build a splat with `sh[2] = (1, 0, 0)` → red channel = SH_C1 *
    // (normalised ray.z); all other channels / bands zero.
    let mut sh = [Vec3::ZERO; SH_BAND_2_COUNT];
    sh[2] = Vec3::new(1.0, 0.0, 0.0);
    let splat = GaussianSplatSh3d::from_sh(
        Vec3::ZERO,
        Vec3::new(1.0, 1.0, 1.0),
        Quat::IDENTITY,
        sh,
        1.0, // opaque so the rendered colour ≈ SH-evaluated colour.
    );

    // Ray 1: along +Z (z=1) → expected red = SH_C1.
    // Ray 2: along +X (z=0) → expected red = 0.
    let origins = vec![Vec3::new(0.0, 0.0, -5.0), Vec3::new(-5.0, 0.0, 0.0)];
    let dirs = vec![Vec3::Z, Vec3::X];

    let (rgba, _depth) = run_forward_sh(&engine, &[splat], &origins, &dirs, 100.0, 1.0 / 255.0);

    // CPU oracle predicts these.
    let basis_pos_z = eval_sh(&sh, Vec3::Z);
    let basis_pos_x = eval_sh(&sh, Vec3::X);
    // For an opaque centred ray, rendered colour = basis * opacity.
    let expected_z = basis_pos_z; // opacity 1
    let expected_x = basis_pos_x;
    let drz = (rgba[0] - expected_z).length();
    let drx = (rgba[1] - expected_x).length();
    assert!(
        drz < 5e-3,
        "+Z ray: expected {expected_z:?}, got {:?} (Δ = {drz})",
        rgba[0]
    );
    assert!(
        drx < 5e-3,
        "+X ray: expected {expected_x:?}, got {:?} (Δ = {drx})",
        rgba[1]
    );

    // Non-zero gradient with viewing angle: rendered red along +Z is
    // markedly different from rendered red along +X.
    let dgradient = (rgba[0].x - rgba[1].x).abs();
    assert!(
        dgradient > 0.1,
        "expected non-trivial view-dependent gradient, got |Δred| = {dgradient}"
    );
    // And it should be ≈ SH_C1.
    assert!(
        (rgba[0].x - SH_C1).abs() < 5e-3,
        "expected red ≈ SH_C1 = {SH_C1}, got {}",
        rgba[0].x
    );
}
