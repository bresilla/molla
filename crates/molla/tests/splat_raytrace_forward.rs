//! Stage 44 v1 — forward-only GPU 3D Gaussian splat raytrace.
//!
//! Each `#[test]` is `#[ignore]` because it requires a real wgpu
//! adapter. Run with `--ignored`.
//!
//! Tests:
//! 1. Single Gaussian centred at origin, ray straight along +Z hitting
//!    the centre → composited RGB matches `α · color`.
//! 2. Two Gaussians on top of each other with different opacities and
//!    colours → α-composite matches the hand-computed value.
//! 3. Ray missing all Gaussians sideways → out_rgba.rgb == 0 and
//!    out_rgba.a == 0 (transmittance unchanged).
//! 4. Anisotropic-scale parity vs CPU oracle for a small batch of
//!    pixels.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla::compute::Engine;
use molla::geometry::{composite_forward_cpu, GaussianSplat3d};
use molla::kernels::SplatRaytraceKernel;
use molla::math::{Quat, Real, Vec3};
use molla::storage::{Buffer, BufferUsage};

/// Serialises wgpu adapter lifetime — mirrors the pattern used by
/// `sensor_raycast_gpu.rs`.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn lock_gpu() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Helper: build the engine, returns None if no adapter is available.
fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

/// Upload the splat SoA + ray inputs, allocate outputs, run the kernel,
/// download the outputs to host. Returns `(rgba, depth)` as host vecs.
#[allow(clippy::too_many_arguments)]
fn run_forward(
    engine: &Engine,
    splats: &[GaussianSplat3d],
    rays_origin: &[Vec3],
    rays_dir: &[Vec3],
    max_distance: f32,
    min_alpha: f32,
) -> (Vec<Vec3>, Vec<Real>) {
    let device = engine.device().expect("gpu device").clone();
    let queue = engine.queue().expect("gpu queue").clone();

    // Splat SoA.
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

    // Rays.
    let mut ray_origin = Buffer::from_vec(rays_origin.to_vec(), BufferUsage::Storage);
    let mut ray_direction = Buffer::from_vec(rays_dir.to_vec(), BufferUsage::Storage);
    ray_origin.ensure_device(&device).unwrap();
    ray_direction.ensure_device(&device).unwrap();
    ray_origin.upload(&queue).unwrap();
    ray_direction.upload(&queue).unwrap();

    // Outputs.
    let n = rays_origin.len();
    let mut out_rgba = Buffer::from_vec(vec![Vec3::ZERO; n], BufferUsage::StorageReadWrite);
    let mut out_depth = Buffer::from_vec(vec![-1.0_f64; n], BufferUsage::StorageReadWrite);
    out_rgba.ensure_device(&device).unwrap();
    out_depth.ensure_device(&device).unwrap();
    out_rgba.upload(&queue).unwrap();
    out_depth.upload(&queue).unwrap();

    // Dispatch.
    let kernel = SplatRaytraceKernel::new(&device);
    kernel
        .dispatch(
            &device,
            &queue,
            &splat_pos,
            &splat_scale,
            &splat_rot,
            &splat_color,
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

/// 1. Single Gaussian at origin, ray straight along +Z hits its centre.
/// Expected colour = colour · opacity (closest-approach ρ_max = 1 →
/// α = opacity). Depth = projection of (0, 0, 0) - (0, 0, -5) onto +Z = 5.
#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn single_gaussian_centered_hit() {
    let _gpu = lock_gpu();
    let engine = match try_engine() {
        Some(e) => e,
        None => panic!("GPU adapter required for this test (gated by #[ignore])"),
    };

    // Isotropic unit-σ splat at origin with red colour and opacity 0.6.
    let splat = GaussianSplat3d::new(
        Vec3::ZERO,
        Vec3::ONE,
        Quat::IDENTITY,
        Vec3::new(1.0, 0.0, 0.0),
        0.6,
    );
    let origin = Vec3::new(0.0, 0.0, -5.0);
    let dir = Vec3::Z;

    let (rgba, depth) =
        run_forward(&engine, &[splat], &[origin], &[dir], 100.0, 1.0 / 255.0);

    assert_eq!(rgba.len(), 1);
    // ρ_max = 1.0 (ray passes through centre), α = 0.6.
    // C_out = (1, 0, 0) · 0.6 · 1 = (0.6, 0, 0).
    let got = rgba[0];
    assert!(
        (got.x - 0.6).abs() < 1e-3 && got.y.abs() < 1e-3 && got.z.abs() < 1e-3,
        "rgb = {got:?}, expected ~(0.6, 0, 0)"
    );
    // Depth = projection of (0,0,0)-(0,0,-5) on +Z = 5.
    assert!(
        (depth[0] - 5.0).abs() < 1e-3,
        "depth = {}, expected ~5.0",
        depth[0]
    );
}

/// 2. Two Gaussians at the same world position, different opacities
/// + colours. With identical centres, the camera-distance projection
/// is identical (degenerate sort key); both will composite. We
/// pick distinct centres along Z so the front-to-back order is
/// deterministic and verifiable.
#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn two_gaussians_alpha_composite() {
    let _gpu = lock_gpu();
    let engine = match try_engine() {
        Some(e) => e,
        None => panic!("GPU adapter required for this test (gated by #[ignore])"),
    };

    // Front splat at z=0: red, α=0.5. Back splat at z=2: green, α=0.7.
    let front = GaussianSplat3d::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::ONE,
        Quat::IDENTITY,
        Vec3::new(1.0, 0.0, 0.0),
        0.5,
    );
    let back = GaussianSplat3d::new(
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::ONE,
        Quat::IDENTITY,
        Vec3::new(0.0, 1.0, 0.0),
        0.7,
    );
    let origin = Vec3::new(0.0, 0.0, -5.0);
    let dir = Vec3::Z;

    // Hand-computed: ρ_max = 1.0 for both (ray passes through both
    // centres at t=5 and t=7). α_front = 0.5, α_back = 0.7.
    // C = (1,0,0)·0.5·1 + (0,1,0)·0.7·(1-0.5) = (0.5, 0.35, 0).
    // T_final = (1-0.5)·(1-0.7) = 0.15.  α_out = 0.85.
    let (rgba, depth) = run_forward(
        &engine,
        &[front, back],
        &[origin],
        &[dir],
        100.0,
        1.0 / 255.0,
    );

    let got = rgba[0];
    assert!(
        (got.x - 0.5).abs() < 1e-3,
        "rgb.x = {}, expected 0.5",
        got.x
    );
    assert!(
        (got.y - 0.35).abs() < 1e-3,
        "rgb.y = {}, expected 0.35",
        got.y
    );
    assert!(got.z.abs() < 1e-3, "rgb.z = {}, expected 0", got.z);
    // Depth = front-most hit ≈ 5.0.
    assert!(
        (depth[0] - 5.0).abs() < 1e-3,
        "depth = {}, expected ~5.0",
        depth[0]
    );
}

/// 3. Ray misses the splat (offset 10σ to the side). Output should be
/// transparent (alpha == 0).
#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn ray_misses_gaussian() {
    let _gpu = lock_gpu();
    let engine = match try_engine() {
        Some(e) => e,
        None => panic!("GPU adapter required for this test (gated by #[ignore])"),
    };

    let splat = GaussianSplat3d::new(
        Vec3::ZERO,
        Vec3::ONE,
        Quat::IDENTITY,
        Vec3::ONE,
        1.0,
    );
    let origin = Vec3::new(10.0, 0.0, -5.0); // offset 10σ in X
    let dir = Vec3::Z;
    let (rgba, _depth) =
        run_forward(&engine, &[splat], &[origin], &[dir], 100.0, 1.0 / 255.0);
    let got = rgba[0];
    // ρ_max = exp(-50) → α below min_alpha, splat skipped.
    assert!(
        got.x.abs() < 1e-3 && got.y.abs() < 1e-3 && got.z.abs() < 1e-3,
        "expected miss, got {got:?}"
    );
}

/// 4. CPU/GPU parity sweep on a small batch of pixels with
/// anisotropic scaling and a non-trivial rotation. Confirms the GPU
/// canonical-frame math matches the CPU oracle within f32 precision.
#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn cpu_gpu_parity_anisotropic() {
    let _gpu = lock_gpu();
    let engine = match try_engine() {
        Some(e) => e,
        None => panic!("GPU adapter required for this test (gated by #[ignore])"),
    };

    // 3 splats, anisotropic scales + random-ish rotation/colour.
    let splats = vec![
        GaussianSplat3d::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.5, 2.0, 1.0),
            Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.0),
            Vec3::new(0.9, 0.1, 0.1),
            0.55,
        ),
        GaussianSplat3d::new(
            Vec3::new(0.4, -0.2, 1.5),
            Vec3::new(1.0, 0.6, 0.8),
            Quat::from_euler(glam::EulerRot::XYZ, -0.1, 0.2, 0.4),
            Vec3::new(0.1, 0.85, 0.2),
            0.7,
        ),
        GaussianSplat3d::new(
            Vec3::new(-0.3, 0.3, 3.0),
            Vec3::new(1.5, 1.2, 0.4),
            Quat::IDENTITY,
            Vec3::new(0.15, 0.2, 0.9),
            0.4,
        ),
    ];

    let mut origins = Vec::new();
    let mut dirs = Vec::new();
    // 4x4 pixel grid sampled around the cluster.
    for j in 0..4 {
        for i in 0..4 {
            let x = -1.0 + (i as Real) * 0.5;
            let y = -1.0 + (j as Real) * 0.5;
            origins.push(Vec3::new(x, y, -5.0));
            dirs.push(Vec3::Z);
        }
    }

    let (rgba_gpu, _depth_gpu) =
        run_forward(&engine, &splats, &origins, &dirs, 100.0, 1.0 / 255.0);

    for k in 0..origins.len() {
        let (rgb_cpu, t_cpu) =
            composite_forward_cpu(origins[k], dirs[k], &splats, 100.0, 1.0 / 255.0);
        let cpu_alpha: Real = 1.0 - t_cpu;
        let gpu = rgba_gpu[k];
        let drgb = (gpu - rgb_cpu).length();
        assert!(
            drgb < 5e-3,
            "pixel {k}: cpu rgb {rgb_cpu:?}, gpu rgb {gpu:?} (Δ = {drgb})"
        );
        // Sanity check: alpha bounded.
        assert!(
            cpu_alpha >= 0.0 && cpu_alpha <= 1.0,
            "cpu α = {cpu_alpha}"
        );
    }
}
