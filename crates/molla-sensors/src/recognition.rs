//! Camera recognition: the objects a camera sees, with their image box and
//! 3D position, from a camera's shape-index and depth channels.
//!
//! Pixel rays follow the camera trace convention: camera axes with -Z
//! forward and +Y up, row 0 at the top, and depth measured along each
//! normalised pixel ray. The host decides what counts as one object by
//! mapping shape indices to object ids (e.g. per body or per entity).

use molla_math::{Real, Vec3};

/// Shape index the camera writes where a ray hits nothing.
pub const NO_SHAPE: u32 = u32::MAX;

/// One recognised object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecognizedObject {
    pub id: u64,
    pub pixels: u32,
    /// Inclusive image box: left, top, right, bottom pixel.
    pub bbox: [u32; 4],
    /// Mean visible surface point in camera axes (m).
    pub position: Vec3,
    /// Extent of the visible surface points along camera axes (m).
    pub size: Vec3,
}

/// Camera-axis direction of a pixel's ray.
pub fn pixel_ray(width: u32, height: u32, fov_y: Real, px: u32, py: u32) -> Vec3 {
    let u = 2.0 * (px as Real + 0.5) / width as Real - 1.0;
    let v = 1.0 - 2.0 * (py as Real + 0.5) / height as Real;
    let tan_v = (0.5 * fov_y).tan();
    let tan_h = tan_v * width as Real / height as Real;
    Vec3::new(u * tan_h, v * tan_v, -1.0).normalize()
}

/// Group a frame's pixels by object and summarise each object seen by at
/// least `min_pixels` pixels, ordered by object id.
pub fn recognize(
    width: u32,
    height: u32,
    fov_y: Real,
    depth: &[Real],
    shape: &[u32],
    object_of: &dyn Fn(u32) -> Option<u64>,
    min_pixels: u32,
) -> Vec<RecognizedObject> {
    struct Acc {
        pixels: u32,
        bbox: [u32; 4],
        sum: Vec3,
        lo: Vec3,
        hi: Vec3,
    }
    let mut objects: std::collections::BTreeMap<u64, Acc> = Default::default();
    let pixels = (width as usize * height as usize)
        .min(depth.len())
        .min(shape.len());
    for index in 0..pixels {
        let (s, d) = (shape[index], depth[index]);
        if s == NO_SHAPE || !d.is_finite() {
            continue;
        }
        let Some(id) = object_of(s) else { continue };
        let (px, py) = (index as u32 % width, index as u32 / width);
        let point = pixel_ray(width, height, fov_y, px, py) * d;
        let acc = objects.entry(id).or_insert(Acc {
            pixels: 0,
            bbox: [px, py, px, py],
            sum: Vec3::ZERO,
            lo: point,
            hi: point,
        });
        acc.pixels += 1;
        acc.bbox = [
            acc.bbox[0].min(px),
            acc.bbox[1].min(py),
            acc.bbox[2].max(px),
            acc.bbox[3].max(py),
        ];
        acc.sum += point;
        acc.lo = acc.lo.min(point);
        acc.hi = acc.hi.max(point);
    }
    objects
        .into_iter()
        .filter(|(_, acc)| acc.pixels >= min_pixels.max(1))
        .map(|(id, acc)| RecognizedObject {
            id,
            pixels: acc.pixels,
            bbox: acc.bbox,
            position: acc.sum / acc.pixels as Real,
            size: acc.hi - acc.lo,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centre_pixel_looks_down_minus_z_and_rows_start_at_the_top() {
        let centre = pixel_ray(3, 3, 1.0, 1, 1);
        assert!((centre - Vec3::NEG_Z).length() < 1e-12);
        assert!(pixel_ray(3, 3, 1.0, 1, 0).y > 0.0);
        assert!(pixel_ray(3, 3, 1.0, 2, 1).x > 0.0);
    }

    #[test]
    fn objects_get_boxes_positions_and_ignore_misses_and_tiny_blobs() {
        let (w, h) = (4u32, 3u32);
        let mut shape = vec![NO_SHAPE; 12];
        let mut depth = vec![Real::INFINITY; 12];
        for (x, y) in [(1, 0), (2, 0), (1, 1), (2, 1)] {
            shape[y * 4 + x] = 7;
            depth[y * 4 + x] = 5.0;
        }
        shape[11] = 9;
        depth[11] = 2.0;
        shape[8] = 3;
        depth[8] = 1.0;
        let object_of = |s: u32| (s != 3).then_some(u64::from(s) * 10);
        let seen = recognize(w, h, 1.0, &depth, &shape, &object_of, 1);
        assert_eq!(seen.iter().map(|o| o.id).collect::<Vec<_>>(), [70, 90]);
        let wall = seen[0];
        assert_eq!((wall.pixels, wall.bbox), (4, [1, 0, 2, 1]));
        assert!(wall.position.z < -4.5 && wall.position.x.abs() < 1e-12 && wall.position.y > 0.0);
        assert!(wall.size.x > 0.0 && wall.size.y > 0.0);
        assert_eq!(recognize(w, h, 1.0, &depth, &shape, &object_of, 2).len(), 1);
    }
}
