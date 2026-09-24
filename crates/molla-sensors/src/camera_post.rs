//! CPU reference and convenience operations on packed RGBA8 images.
//!
//! Each operation unpacks normalized RGB, adjusts it and quantizes back to RGBA8.
//! Alpha is preserved. Color-space conversion is explicit; camera color starts
//! in linear space. [`crate::ImageProcessor`] executes corresponding device-only
//! pipelines with per-image white-balance reductions.

#![cfg(feature = "gpu")]

/// Unpack `0xAABBGGRR` (the `u32` layout `TiledCamera::download_color`
/// returns) into normalized RGBA in `[0, 1]`, without color-space conversion.
pub fn unpack_rgba8(p: u32) -> [f32; 4] {
    let r = ((p & 0xFF) as f32) / 255.0;
    let g = (((p >> 8) & 0xFF) as f32) / 255.0;
    let b = (((p >> 16) & 0xFF) as f32) / 255.0;
    let a = (((p >> 24) & 0xFF) as f32) / 255.0;
    [r, g, b, a]
}

/// Pack a linear `[r, g, b, a]` in `[0, 1]` back into the
/// `0xAABBGGRR` `u32` `TiledCamera` writes. Out-of-range values are
/// clamped; NaN/Inf survive `clamp` only on stable channels, so the
/// helper substitutes 0 explicitly to avoid undefined `as u32` casts
/// from non-finite floats.
pub fn pack_rgba8(rgba: [f32; 4]) -> u32 {
    let q = |v: f32| {
        if !v.is_finite() {
            0u32
        } else {
            (v.clamp(0.0, 1.0) * 255.0).round() as u32
        }
    };
    q(rgba[0]) | (q(rgba[1]) << 8) | (q(rgba[2]) << 16) | (q(rgba[3]) << 24)
}

/// sRGB → linear (per-channel). Standard piecewise formula.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear → sRGB (per-channel). Standard piecewise formula.
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Multiply RGB by `exposure`. Per-pixel in place; `exposure > 1`
/// brightens, `< 1` darkens.
pub fn apply_exposure(buf: &mut [u32], exposure: f32) {
    for p in buf.iter_mut() {
        let mut c = unpack_rgba8(*p);
        c[0] *= exposure;
        c[1] *= exposure;
        c[2] *= exposure;
        *p = pack_rgba8(c);
    }
}

/// Pixel-wise sRGB gamma compression (linear `[0,1]` → sRGB-encoded
/// `[0,1]`). Use this when the renderer wrote linear radiance.
pub fn apply_gamma_srgb(buf: &mut [u32]) {
    for p in buf.iter_mut() {
        let c = unpack_rgba8(*p);
        *p = pack_rgba8([
            linear_to_srgb(c[0]),
            linear_to_srgb(c[1]),
            linear_to_srgb(c[2]),
            c[3],
        ]);
    }
}

/// Decode sRGB-encoded RGB into linear RGB, preserving alpha.
pub fn apply_srgb_to_linear(buf: &mut [u32]) {
    for p in buf.iter_mut() {
        let c = unpack_rgba8(*p);
        *p = pack_rgba8([
            srgb_to_linear(c[0]),
            srgb_to_linear(c[1]),
            srgb_to_linear(c[2]),
            c[3],
        ]);
    }
}

/// Generic exponent gamma compression `c' = c^(1/γ)`. `γ = 2.2` is a
/// rough sRGB approximation; the proper [`apply_gamma_srgb`] uses
/// the piecewise formula.
pub fn apply_gamma(buf: &mut [u32], gamma: f32) {
    let inv = 1.0 / gamma.max(1.0e-6);
    for p in buf.iter_mut() {
        let c = unpack_rgba8(*p);
        *p = pack_rgba8([
            c[0].max(0.0).powf(inv),
            c[1].max(0.0).powf(inv),
            c[2].max(0.0).powf(inv),
            c[3],
        ]);
    }
}

/// Reinhard tone mapping `c' = c / (1 + c)`. Compresses unbounded
/// linear radiance into `[0, 1)`; combined with [`apply_gamma_srgb`]
/// gives a "photograph-looking" output.
pub fn apply_reinhard_tone_map(buf: &mut [u32]) {
    for p in buf.iter_mut() {
        let c = unpack_rgba8(*p);
        *p = pack_rgba8([
            c[0] / (1.0 + c[0]),
            c[1] / (1.0 + c[1]),
            c[2] / (1.0 + c[2]),
            c[3],
        ]);
    }
}

/// Reinhard tone mapping with a white-point `Lwhite`:
/// `c' = c · (1 + c/Lw²) / (1 + c)`. Lets highlights above
/// `Lwhite` still saturate to white.
pub fn apply_reinhard_white_point(buf: &mut [u32], lwhite: f32) {
    let lw2 = lwhite.max(1.0e-6).powi(2);
    for p in buf.iter_mut() {
        let c = unpack_rgba8(*p);
        let f = |x: f32| x * (1.0 + x / lw2) / (1.0 + x);
        *p = pack_rgba8([f(c[0]), f(c[1]), f(c[2]), c[3]]);
    }
}

/// Gray-world white balance: scale each channel so its mean equals
/// the overall mean luminance. Returns the per-channel scale factors
/// applied so callers can chain.
pub fn apply_gray_world_wb(buf: &mut [u32]) -> [f32; 3] {
    if buf.is_empty() {
        return [1.0, 1.0, 1.0];
    }
    let mut sum = [0.0f64; 3];
    for p in buf.iter() {
        let c = unpack_rgba8(*p);
        sum[0] += c[0] as f64;
        sum[1] += c[1] as f64;
        sum[2] += c[2] as f64;
    }
    let n = buf.len() as f64;
    let mean = [sum[0] / n, sum[1] / n, sum[2] / n];
    let gray = (mean[0] + mean[1] + mean[2]) / 3.0;
    let scale = [
        (gray / mean[0].max(1.0e-9)) as f32,
        (gray / mean[1].max(1.0e-9)) as f32,
        (gray / mean[2].max(1.0e-9)) as f32,
    ];
    for p in buf.iter_mut() {
        let c = unpack_rgba8(*p);
        *p = pack_rgba8([c[0] * scale[0], c[1] * scale[1], c[2] * scale[2], c[3]]);
    }
    scale
}

/// White-patch white balance: find each channel's maximum and scale
/// it to `1.0`. Returns the scale factors used.
pub fn apply_white_patch_wb(buf: &mut [u32]) -> [f32; 3] {
    let mut max = [0.0f32; 3];
    for p in buf.iter() {
        let c = unpack_rgba8(*p);
        max[0] = max[0].max(c[0]);
        max[1] = max[1].max(c[1]);
        max[2] = max[2].max(c[2]);
    }
    let scale = [
        (1.0 / max[0].max(1.0e-9)).min(1.0e6),
        (1.0 / max[1].max(1.0e-9)).min(1.0e6),
        (1.0 / max[2].max(1.0e-9)).min(1.0e6),
    ];
    for p in buf.iter_mut() {
        let c = unpack_rgba8(*p);
        *p = pack_rgba8([c[0] * scale[0], c[1] * scale[1], c[2] * scale[2], c[3]]);
    }
    scale
}

/// Brightness + contrast: `c' = (c − 0.5) · contrast + 0.5 + brightness`.
pub fn apply_brightness_contrast(buf: &mut [u32], brightness: f32, contrast: f32) {
    for p in buf.iter_mut() {
        let c = unpack_rgba8(*p);
        let f = |x: f32| (x - 0.5) * contrast + 0.5 + brightness;
        *p = pack_rgba8([f(c[0]), f(c[1]), f(c[2]), c[3]]);
    }
}

/// Saturation: blend toward / away from grayscale (luminance-coded).
/// `s = 0` returns full grayscale; `s = 1` is identity; `s > 1`
/// boosts saturation.
pub fn apply_saturation(buf: &mut [u32], saturation: f32) {
    for p in buf.iter_mut() {
        let c = unpack_rgba8(*p);
        let l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let f = |x: f32| l + (x - l) * saturation;
        *p = pack_rgba8([f(c[0]), f(c[1]), f(c[2]), c[3]]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba8_round_trips_through_pack_unpack() {
        let original = 0x80_60_40_20u32; // A=128, B=96, G=64, R=32
        let c = unpack_rgba8(original);
        let back = pack_rgba8(c);
        assert_eq!(back, original);
    }

    #[test]
    fn srgb_round_trip_is_close_to_identity() {
        for &c in &[0.0_f32, 0.1, 0.3, 0.5, 0.8, 1.0] {
            let r = linear_to_srgb(srgb_to_linear(c));
            assert!((r - c).abs() < 1.0e-5, "round trip {c} -> {r}");
        }
    }

    #[test]
    fn exposure_doubles_brightness() {
        let mut buf = vec![pack_rgba8([0.25, 0.25, 0.25, 1.0])];
        apply_exposure(&mut buf, 2.0);
        let c = unpack_rgba8(buf[0]);
        assert!((c[0] - 0.5).abs() < 1.0e-2);
        assert!((c[1] - 0.5).abs() < 1.0e-2);
        assert!((c[2] - 0.5).abs() < 1.0e-2);
    }

    #[test]
    fn gray_world_wb_neutralises_a_tinted_uniform_image() {
        // Uniform image with a red-heavy tint.
        let mut buf = vec![pack_rgba8([0.8, 0.4, 0.4, 1.0]); 16];
        apply_gray_world_wb(&mut buf);
        let c = unpack_rgba8(buf[0]);
        // After gray-world the three channels should be ~ equal.
        assert!((c[0] - c[1]).abs() < 0.02);
        assert!((c[1] - c[2]).abs() < 0.02);
    }

    #[test]
    fn reinhard_clamps_unbounded_input_into_unit_interval() {
        let mut buf = vec![pack_rgba8([1.0, 1.0, 1.0, 1.0])];
        // Reinhard maps c → c/(1+c); for c = 1.0 → 0.5.
        apply_reinhard_tone_map(&mut buf);
        let c = unpack_rgba8(buf[0]);
        assert!((c[0] - 0.5).abs() < 1.0e-2);
    }

    #[test]
    fn saturation_zero_collapses_to_grayscale() {
        let mut buf = vec![pack_rgba8([0.8, 0.1, 0.1, 1.0])];
        apply_saturation(&mut buf, 0.0);
        let c = unpack_rgba8(buf[0]);
        // All three channels should equal the luminance.
        assert!((c[0] - c[1]).abs() < 1.0e-2);
        assert!((c[1] - c[2]).abs() < 1.0e-2);
    }

    #[test]
    fn nan_input_packs_to_zero_rather_than_undefined() {
        // Any pipeline that produces NaN (e.g. exposure on an
        // already-clipped buffer with bad inputs) must round-trip to
        // a defined pixel rather than rely on the UB of `as u32`.
        let bad = pack_rgba8([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0]);
        let back = unpack_rgba8(bad);
        // The three corrupted channels collapse to 0; alpha stays.
        assert_eq!(back[0], 0.0);
        assert_eq!(back[1], 0.0);
        assert_eq!(back[2], 0.0);
        assert!((back[3] - 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn exposure_pair_is_near_identity_in_unsaturated_range() {
        // exposure(0.5) ∘ exposure(2.0) ≈ identity provided the
        // first pass doesn't saturate any channel. Picked starting
        // values whose ×2 stays below 1.0; 8-bit quantization gives
        // ~±1 LSB ≈ ±0.004 noise.
        let original = pack_rgba8([0.1, 0.2, 0.35, 1.0]);
        let mut buf = vec![original];
        apply_exposure(&mut buf, 2.0);
        apply_exposure(&mut buf, 0.5);
        let c = unpack_rgba8(buf[0]);
        let o = unpack_rgba8(original);
        for k in 0..3 {
            let diff = (c[k] - o[k]).abs();
            assert!(diff < 1.0e-2, "channel {k}: |{} − {}| = {diff}", c[k], o[k]);
        }
    }

    #[test]
    fn gamma_pair_is_near_identity() {
        let original = pack_rgba8([0.4, 0.5, 0.6, 1.0]);
        let mut buf = vec![original];
        apply_gamma(&mut buf, 2.2);
        apply_gamma(&mut buf, 1.0 / 2.2);
        let c = unpack_rgba8(buf[0]);
        let o = unpack_rgba8(original);
        for k in 0..3 {
            assert!((c[k] - o[k]).abs() < 1.0e-2);
        }
    }

    #[test]
    fn empty_buffer_pipeline_is_safe() {
        let mut empty: Vec<u32> = Vec::new();
        apply_exposure(&mut empty, 2.0);
        apply_reinhard_tone_map(&mut empty);
        apply_gamma(&mut empty, 2.2);
        let scale = apply_gray_world_wb(&mut empty);
        assert_eq!(scale, [1.0, 1.0, 1.0]);
        let scale = apply_white_patch_wb(&mut empty);
        // All-zero max → 1.0 / ε, clamped at 1e6.
        for s in scale {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn brightness_contrast_identity_when_one_and_zero() {
        let original = pack_rgba8([0.3, 0.5, 0.7, 1.0]);
        let mut buf = vec![original];
        apply_brightness_contrast(&mut buf, 0.0, 1.0);
        let c = unpack_rgba8(buf[0]);
        let o = unpack_rgba8(original);
        for k in 0..3 {
            assert!((c[k] - o[k]).abs() < 1.0e-2);
        }
    }
}
