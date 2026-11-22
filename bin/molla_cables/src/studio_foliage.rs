//! GPU texture arrays for the baked ambientCG leaf and bark scans.
use crate::foliage_layout as layout;

use eframe::wgpu;
use image::{RgbaImage, imageops};

const LEAF_COLOR: &[u8] = include_bytes!("../data/foliage/leaves-color.jpg");
const LEAF_ALPHA: &[u8] = include_bytes!("../data/foliage/leaves-alpha.png");
const LEAF_NORMAL: &[u8] = include_bytes!("../data/foliage/leaves-normal.jpg");
const BARK_COLOR: &[u8] = include_bytes!("../data/foliage/bark-color.jpg");
const BARK_NORMAL: &[u8] = include_bytes!("../data/foliage/bark-normal.jpg");

pub struct Textures {
    pub leaf_color: wgpu::TextureView,
    pub leaf_normal: wgpu::TextureView,
    pub bark_color: wgpu::TextureView,
    pub bark_normal: wgpu::TextureView,
}

fn decode(bytes: &[u8]) -> RgbaImage {
    image::load_from_memory(bytes)
        .expect("embedded foliage texture decodes")
        .to_rgba8()
}

fn split(atlas: &RgbaImage, [w, h]: [u32; 2], columns: u32, count: usize) -> Vec<RgbaImage> {
    (0..count as u32)
        .map(|i| imageops::crop_imm(atlas, (i % columns) * w, (i / columns) * h, w, h).to_image())
        .collect()
}

fn coverage(image: &RgbaImage, scale: f64) -> f64 {
    let covered = image
        .pixels()
        .filter(|p| f64::from(p.0[3]) * scale > 127.5)
        .count();
    covered as f64 / (image.width() * image.height()) as f64
}

/// Rescales alpha so the fraction of texels passing the 0.5 test matches `target`.
fn preserve_coverage(image: &mut RgbaImage, target: f64) {
    let (mut low, mut high) = (0.25, 16.0);
    for _ in 0..24 {
        let mid = 0.5 * (low + high);
        if coverage(image, mid) < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    let scale = 0.5 * (low + high);
    for p in image.pixels_mut() {
        p.0[3] = (f64::from(p.0[3]) * scale).round().min(255.0) as u8;
    }
}

fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    layers: Vec<RgbaImage>,
    format: wgpu::TextureFormat,
    alpha_tested: bool,
) -> wgpu::TextureView {
    let (width, height) = layers[0].dimensions();
    let levels = width.max(height).ilog2() + 1;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: layers.len() as u32,
        },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (layer, mut image) in layers.into_iter().enumerate() {
        let target = coverage(&image, 1.0);
        for level in 0..levels {
            let (w, h) = image.dimensions();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                image.as_raw(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            image = imageops::resize(
                &image,
                (w / 2).max(1),
                (h / 2).max(1),
                imageops::FilterType::Triangle,
            );
            if alpha_tested {
                preserve_coverage(&mut image, target);
            }
        }
    }
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

pub fn load(device: &wgpu::Device, queue: &wgpu::Queue) -> Textures {
    let mut leaf_color = decode(LEAF_COLOR);
    let alpha = image::load_from_memory(LEAF_ALPHA)
        .expect("embedded leaf alpha decodes")
        .to_luma8();
    for (p, a) in leaf_color.pixels_mut().zip(alpha.pixels()) {
        p.0[3] = a.0[0];
    }
    let leaves = |atlas: &RgbaImage| {
        split(
            atlas,
            layout::LEAF_CELL,
            layout::LEAF_COLUMNS,
            layout::LEAF_TEXTURES,
        )
    };
    let bark = |atlas: &RgbaImage| {
        let size = layout::BARK_CELL;
        split(atlas, [size, size], 1, layout::BARK_SETS.len())
    };
    let srgb = wgpu::TextureFormat::Rgba8UnormSrgb;
    let linear = wgpu::TextureFormat::Rgba8Unorm;
    Textures {
        leaf_color: upload(device, queue, "leaf albedo", leaves(&leaf_color), srgb, true),
        leaf_normal: upload(
            device,
            queue,
            "leaf normals",
            leaves(&decode(LEAF_NORMAL)),
            linear,
            false,
        ),
        bark_color: upload(device, queue, "bark albedo", bark(&decode(BARK_COLOR)), srgb, false),
        bark_normal: upload(
            device,
            queue,
            "bark normals",
            bark(&decode(BARK_NORMAL)),
            linear,
            false,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_foliage_matches_the_layout_and_keeps_alpha_coverage_through_mips() {
        let color = decode(LEAF_COLOR);
        let [w, h] = layout::LEAF_CELL;
        let rows = (layout::LEAF_TEXTURES as u32).div_ceil(layout::LEAF_COLUMNS);
        assert_eq!(color.dimensions(), (w * layout::LEAF_COLUMNS, h * rows));
        let alpha = image::load_from_memory(LEAF_ALPHA).unwrap().to_luma8();
        assert_eq!(alpha.dimensions(), color.dimensions());
        let bark = decode(BARK_COLOR);
        let size = layout::BARK_CELL;
        assert_eq!(bark.dimensions(), (size, size * layout::BARK_SETS.len() as u32));
        let mut atlas = color;
        for (p, a) in atlas.pixels_mut().zip(alpha.pixels()) {
            p.0[3] = a.0[0];
        }
        for leaf in split(&atlas, [w, h], layout::LEAF_COLUMNS, layout::LEAF_TEXTURES) {
            let target = coverage(&leaf, 1.0);
            assert!((0.35..0.7).contains(&target), "leaf coverage {target}");
            let mut small = imageops::resize(&leaf, w / 8, h / 8, imageops::FilterType::Triangle);
            preserve_coverage(&mut small, target);
            assert!((coverage(&small, 1.0) - target).abs() < 0.03);
        }
    }
}
