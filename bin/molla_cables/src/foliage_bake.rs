//! Bakes ambientCG leaf atlases and bark materials into Molla's packed foliage textures.
//!
//! Usage: `molla_foliage_bake <extracted-ambientcg-dir> <output-dir>`.
mod foliage_layout;

use foliage_layout as layout;
use image::{GrayImage, Luma, Rgb, RgbImage, imageops};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

const ALONG_BINS: usize = 384;
const SUPERSAMPLE: usize = 3;
const BLEED_PASSES: usize = 24;

struct Source {
    color: RgbImage,
    opacity: GrayImage,
    normal: RgbImage,
}

struct Leaf {
    color: RgbImage,
    alpha: GrayImage,
    normal: RgbImage,
}

/// Principal frame and smoothed silhouette envelope of one extracted leaf.
struct Blade {
    center: [f64; 2],
    up: [f64; 2],
    right: [f64; 2],
    base: f64,
    tip: f64,
    left: Vec<f64>,
    right_edge: Vec<f64>,
}

fn load(dir: &Path, id: &str, map: &str) -> Result<image::DynamicImage> {
    let path = dir.join(format!("{id}_2K-JPG")).join(format!("{id}_2K-JPG_{map}.jpg"));
    image::open(&path).map_err(|e| format!("{}: {e}", path.display()))
}

fn bilinear(image: &RgbImage, x: f64, y: f64) -> [f64; 3] {
    let (w, h) = image.dimensions();
    let x = x.clamp(0.0, w as f64 - 1.001);
    let y = y.clamp(0.0, h as f64 - 1.001);
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let p = |dx, dy| image.get_pixel(x0 + dx, y0 + dy).0.map(f64::from);
    let [a, b, c, d] = [p(0, 0), p(1, 0), p(0, 1), p(1, 1)];
    std::array::from_fn(|i| {
        (a[i] * (1.0 - fx) + b[i] * fx) * (1.0 - fy) + (c[i] * (1.0 - fx) + d[i] * fx) * fy
    })
}

fn bilinear_masked(opacity: &GrayImage, labels: &[u32], label: u32, x: f64, y: f64) -> f64 {
    let (w, h) = opacity.dimensions();
    if x < 0.0 || y < 0.0 || x >= w as f64 - 1.0 || y >= h as f64 - 1.0 {
        return 0.0;
    }
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let p = |dx: u32, dy: u32| {
        let (px, py) = (x0 + dx, y0 + dy);
        if labels[(py * w + px) as usize] == label {
            f64::from(opacity.get_pixel(px, py).0[0])
        } else {
            0.0
        }
    };
    (p(0, 0) * (1.0 - fx) + p(1, 0) * fx) * (1.0 - fy) + (p(0, 1) * (1.0 - fx) + p(1, 1) * fx) * fy
}

/// Labels 8-connected opaque regions; returns labels and each region's pixel list.
fn components(opacity: &GrayImage) -> (Vec<u32>, Vec<Vec<[u32; 2]>>) {
    let (w, h) = opacity.dimensions();
    let mut labels = vec![u32::MAX; (w * h) as usize];
    let mut regions = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        if labels[start as usize] != u32::MAX || opacity.as_raw()[start as usize] < 128 {
            continue;
        }
        let label = regions.len() as u32;
        let mut pixels = Vec::new();
        labels[start as usize] = label;
        stack.push(start);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            pixels.push([x, y]);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = (ny as u32 * w + nx as u32) as usize;
                    if labels[j] == u32::MAX && opacity.as_raw()[j] >= 128 {
                        labels[j] = label;
                        stack.push(j as u32);
                    }
                }
            }
        }
        regions.push(pixels);
    }
    (labels, regions)
}

fn running_extreme(values: &[f64], radius: usize, max: bool) -> Vec<f64> {
    (0..values.len())
        .map(|i| {
            let window = &values[i.saturating_sub(radius)..(i + radius + 1).min(values.len())];
            let finite = window.iter().copied().filter(|v| v.is_finite());
            if max {
                finite.fold(f64::NEG_INFINITY, f64::max)
            } else {
                finite.fold(f64::INFINITY, f64::min)
            }
        })
        .collect()
}

fn smooth(values: &[f64], radius: usize) -> Vec<f64> {
    (0..values.len())
        .map(|i| {
            let window = &values[i.saturating_sub(radius)..(i + radius + 1).min(values.len())];
            window.iter().sum::<f64>() / window.len() as f64
        })
        .collect()
}

fn blade(pixels: &[[u32; 2]]) -> Result<Blade> {
    let n = pixels.len() as f64;
    let center = [0, 1].map(|k| pixels.iter().map(|p| f64::from(p[k])).sum::<f64>() / n);
    let top = pixels.iter().map(|p| p[1]).min().ok_or("empty leaf")?;
    let apex: Vec<_> = pixels.iter().filter(|p| p[1] <= top + 3).collect();
    let tip = [0, 1].map(|k| apex.iter().map(|p| f64::from(p[k])).sum::<f64>() / apex.len() as f64);
    let (dx, dy) = (tip[0] - center[0], tip[1] - center[1]);
    let length = dx.hypot(dy);
    let up = [dx / length, dy / length];
    if up[1] > -0.7 {
        return Err("leaf is not upright in its atlas".into());
    }
    let right = [-up[1], up[0]];
    let project = |p: &[u32; 2], axis: [f64; 2]| {
        (f64::from(p[0]) - center[0]) * axis[0] + (f64::from(p[1]) - center[1]) * axis[1]
    };
    let along: Vec<f64> = pixels.iter().map(|p| project(p, up)).collect();
    let low = along.iter().copied().fold(f64::INFINITY, f64::min);
    let high = along.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut left = vec![f64::INFINITY; ALONG_BINS];
    let mut right_edge = vec![f64::NEG_INFINITY; ALONG_BINS];
    for (p, a) in pixels.iter().zip(&along) {
        let bin = ((((a - low) / (high - low)) * ALONG_BINS as f64) as usize).min(ALONG_BINS - 1);
        let lateral = project(p, right);
        left[bin] = left[bin].min(lateral);
        right_edge[bin] = right_edge[bin].max(lateral);
    }
    fill(&mut left);
    fill(&mut right_edge);
    let width: Vec<f64> = left.iter().zip(&right_edge).map(|(a, b)| (b - a).max(0.0)).collect();
    let widest = width.iter().copied().fold(0.0, f64::max);
    let start = width.iter().take_while(|&&w| w < 0.15 * widest).count();
    let reach = ALONG_BINS / 40;
    let left = smooth(&running_extreme(&left, reach, false), reach);
    let right_edge = smooth(&running_extreme(&right_edge, reach, true), reach);
    let base = low + (high - low) * start as f64 / ALONG_BINS as f64;
    Ok(Blade {
        center,
        up,
        right,
        base,
        tip: high,
        left: rebin(&left, low, high, base),
        right_edge: rebin(&right_edge, low, high, base),
    })
}

/// Replaces empty profile bins with the nearest filled bin.
fn fill(values: &mut [f64]) {
    let filled: Vec<usize> = (0..values.len()).filter(|&i| values[i].is_finite()).collect();
    for i in 0..values.len() {
        if !values[i].is_finite() {
            let nearest = filled.iter().min_by_key(|&&j| j.abs_diff(i)).copied().unwrap_or(i);
            values[i] = values[nearest];
        }
    }
}

/// Resamples a profile defined over `[low, high]` onto `ALONG_BINS` bins over `[base, high]`.
fn rebin(values: &[f64], low: f64, high: f64, base: f64) -> Vec<f64> {
    (0..ALONG_BINS)
        .map(|i| {
            let a = base + (high - base) * (i as f64 + 0.5) / ALONG_BINS as f64;
            let f = ((a - low) / (high - low) * ALONG_BINS as f64 - 0.5).clamp(0.0, (ALONG_BINS - 1) as f64);
            let i0 = f.floor() as usize;
            let i1 = (i0 + 1).min(ALONG_BINS - 1);
            values[i0] * (1.0 - (f - i0 as f64)) + values[i1] * (f - i0 as f64)
        })
        .collect()
}

impl Blade {
    fn envelope(&self, t: f64) -> (f64, f64) {
        let f = (t * ALONG_BINS as f64 - 0.5).clamp(0.0, (ALONG_BINS - 1) as f64);
        let i0 = f.floor() as usize;
        let i1 = (i0 + 1).min(ALONG_BINS - 1);
        let u = f - i0 as f64;
        (
            self.left[i0] * (1.0 - u) + self.left[i1] * u,
            self.right_edge[i0] * (1.0 - u) + self.right_edge[i1] * u,
        )
    }

    /// Source pixel for blade point `t` along and `q` in [-1, 1] across the outline.
    fn source(&self, t: f64, q: f64) -> [f64; 2] {
        let along = self.base + t * (self.tip - self.base);
        let (left, right) = self.envelope(t);
        let lateral = 0.5 * (left + right) + q * 0.5 * (right - left);
        [0, 1].map(|k| self.center[k] + self.up[k] * along + self.right[k] * lateral)
    }
}

fn extract(source: &Source, labels: &[u32], label: u32, blade: &Blade) -> Leaf {
    let [w, h] = layout::LEAF_CELL;
    let mut color = RgbImage::new(w, h);
    let mut alpha = GrayImage::new(w, h);
    let mut normal = RgbImage::new(w, h);
    let s = SUPERSAMPLE as f64;
    for j in 0..h {
        for i in 0..w {
            let (mut rgb, mut a, mut n) = ([0.0; 3], 0.0, [0.0; 3]);
            for sy in 0..SUPERSAMPLE {
                for sx in 0..SUPERSAMPLE {
                    let u = (i as f64 + (sx as f64 + 0.5) / s) / w as f64;
                    let v = (j as f64 + (sy as f64 + 0.5) / s) / h as f64;
                    let t = 1.0 - v;
                    let x = (2.0 * u - 1.0) * layout::HALF_EXTENT;
                    let q = x / layout::half_width(t);
                    if q.abs() > 1.4 {
                        continue;
                    }
                    let [px, py] = blade.source(t, q);
                    let coverage = bilinear_masked(&source.opacity, labels, label, px, py) / 255.0;
                    let c = bilinear(&source.color, px, py);
                    let m = bilinear(&source.normal, px, py);
                    let gl = [m[0] / 127.5 - 1.0, 1.0 - m[1] / 127.5];
                    let rotated = [
                        gl[0] * blade.right[0] + gl[1] * blade.right[1],
                        gl[0] * blade.up[0] + gl[1] * blade.up[1],
                        m[2] / 127.5 - 1.0,
                    ];
                    for k in 0..3 {
                        rgb[k] += c[k] * coverage;
                        n[k] += rotated[k] * coverage;
                    }
                    a += coverage;
                }
            }
            let samples = s * s;
            alpha.put_pixel(i, j, Luma([(255.0 * a / samples).round() as u8]));
            if a > 1e-6 {
                color.put_pixel(i, j, Rgb(rgb.map(|c| (c / a).round().clamp(0.0, 255.0) as u8)));
                let length = n.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-9);
                normal.put_pixel(
                    i,
                    j,
                    Rgb(n.map(|v| (127.5 * (v / length + 1.0)).round().clamp(0.0, 255.0) as u8)),
                );
            }
        }
    }
    bleed(&mut color, &mut normal, &alpha);
    Leaf {
        color,
        alpha,
        normal,
    }
}

/// Extends edge colors and normals into transparent texels.
fn bleed(color: &mut RgbImage, normal: &mut RgbImage, alpha: &GrayImage) {
    let (w, h) = color.dimensions();
    let mut filled: Vec<bool> = alpha.as_raw().iter().map(|&a| a > 0).collect();
    for _ in 0..BLEED_PASSES {
        let previous = filled.clone();
        for y in 0..h {
            for x in 0..w {
                if previous[(y * w + x) as usize] {
                    continue;
                }
                let (mut c, mut n, mut count) = ([0u32; 3], [0u32; 3], 0);
                for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    if previous[(ny as u32 * w + nx as u32) as usize] {
                        let (cp, np) = (color.get_pixel(nx as u32, ny as u32), normal.get_pixel(nx as u32, ny as u32));
                        for k in 0..3 {
                            c[k] += u32::from(cp.0[k]);
                            n[k] += u32::from(np.0[k]);
                        }
                        count += 1;
                    }
                }
                if count > 0 {
                    color.put_pixel(x, y, Rgb(c.map(|v| (v / count) as u8)));
                    normal.put_pixel(x, y, Rgb(n.map(|v| (v / count) as u8)));
                    filled[(y * w + x) as usize] = true;
                }
            }
        }
    }
    let opaque: Vec<_> = (0..w * h).filter(|&i| filled[i as usize]).collect();
    let mean: [u8; 3] = std::array::from_fn(|k| {
        let sum: u64 = opaque.iter().map(|&i| u64::from(color.as_raw()[(i * 3) as usize + k])).sum();
        (sum / opaque.len().max(1) as u64) as u8
    });
    for i in 0..w * h {
        if !filled[i as usize] {
            color.put_pixel(i % w, i / w, Rgb(mean));
            normal.put_pixel(i % w, i / w, Rgb([128, 128, 255]));
        }
    }
}

fn leaves(dir: &Path, id: &str) -> Result<Vec<Leaf>> {
    let source = Source {
        color: load(dir, id, "Color")?.to_rgb8(),
        opacity: load(dir, id, "Opacity")?.to_luma8(),
        normal: load(dir, id, "NormalGL")?.to_rgb8(),
    };
    let (labels, regions) = components(&source.opacity);
    let (w, h) = source.opacity.dimensions();
    let mut kept: Vec<(u32, &Vec<[u32; 2]>)> = regions
        .iter()
        .enumerate()
        .filter(|(_, p)| p.len() as f64 > 0.004 * f64::from(w * h))
        .map(|(i, p)| (i as u32, p))
        .collect();
    let row = |p: &[[u32; 2]]| p.iter().map(|q| u64::from(q[1])).sum::<u64>() / p.len() as u64 / 256;
    let column = |p: &[[u32; 2]]| p.iter().map(|q| u64::from(q[0])).sum::<u64>() / p.len() as u64;
    kept.sort_by_key(|(_, p)| (row(p), column(p)));
    kept.into_iter()
        .map(|(label, pixels)| {
            let blade = blade(pixels).map_err(|e| format!("{id}: {e}"))?;
            Ok(extract(&source, &labels, label, &blade))
        })
        .collect()
}

fn bark(dir: &Path, id: &str) -> Result<(RgbImage, RgbImage)> {
    let size = layout::BARK_CELL;
    let resize = |image: image::DynamicImage| {
        imageops::resize(&image.to_rgb8(), size, size, imageops::FilterType::Lanczos3)
    };
    Ok((resize(load(dir, id, "Color")?), resize(load(dir, id, "NormalGL")?)))
}

fn save_jpeg(image: &RgbImage, path: &Path) -> Result<()> {
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(file), 92)
        .encode_image(image)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).map(PathBuf::from);
    let usage = "usage: molla_foliage_bake <extracted-ambientcg-dir> <output-dir>";
    let (source, output) = (args.next().ok_or(usage)?, args.next().ok_or(usage)?);
    std::fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let mut all = Vec::new();
    for id in layout::LEAF_SETS {
        let found = leaves(&source, id)?;
        println!("{id}: {} leaves", found.len());
        all.extend(found);
    }
    if all.len() != layout::LEAF_TEXTURES {
        return Err(format!("expected {} leaves, found {}", layout::LEAF_TEXTURES, all.len()));
    }
    let [cw, ch] = layout::LEAF_CELL;
    let rows = (layout::LEAF_TEXTURES as u32).div_ceil(layout::LEAF_COLUMNS);
    let mut color = RgbImage::new(cw * layout::LEAF_COLUMNS, ch * rows);
    let mut alpha = GrayImage::new(cw * layout::LEAF_COLUMNS, ch * rows);
    let mut normal = RgbImage::from_pixel(cw * layout::LEAF_COLUMNS, ch * rows, Rgb([128, 128, 255]));
    for (i, leaf) in all.iter().enumerate() {
        let (x, y) = ((i as u32 % layout::LEAF_COLUMNS) * cw, (i as u32 / layout::LEAF_COLUMNS) * ch);
        imageops::replace(&mut color, &leaf.color, x.into(), y.into());
        imageops::replace(&mut alpha, &leaf.alpha, x.into(), y.into());
        imageops::replace(&mut normal, &leaf.normal, x.into(), y.into());
    }
    save_jpeg(&color, &output.join("leaves-color.jpg"))?;
    save_jpeg(&normal, &output.join("leaves-normal.jpg"))?;
    alpha
        .save(output.join("leaves-alpha.png"))
        .map_err(|e| e.to_string())?;
    let size = layout::BARK_CELL;
    let layers = layout::BARK_SETS.len() as u32;
    let mut bark_color = RgbImage::new(size, size * layers);
    let mut bark_normal = RgbImage::new(size, size * layers);
    for (i, id) in layout::BARK_SETS.iter().enumerate() {
        let (c, n) = bark(&source, id)?;
        imageops::replace(&mut bark_color, &c, 0, i64::from(i as u32 * size));
        imageops::replace(&mut bark_normal, &n, 0, i64::from(i as u32 * size));
    }
    save_jpeg(&bark_color, &output.join("bark-color.jpg"))?;
    save_jpeg(&bark_normal, &output.join("bark-normal.jpg"))?;
    println!("wrote {} leaf and {layers} bark layers to {}", all.len(), output.display());
    Ok(())
}
