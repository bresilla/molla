//! H9 — Synthetic-image annotation export.
//!
//! Builds bounding-box / semantic-class / instance-mask outputs from
//! [`TiledCamera`](crate::TiledCamera)'s `SHAPE_INDEX` channel. The
//! API mirrors helios's `SyntheticAnnotation` plug-in
//! (`xtra/helios/plugins/syntheticannotation/include/SyntheticAnnotation.h`)
//! but operates on molla's existing GPU camera output rather than
//! re-rendering — `TiledCamera` already emits a `u32` shape-id per
//! pixel, so all H9 needs to do is map shape-ids to class-ids and
//! reduce to bounding boxes.
//!
//! Usage shape:
//!
//! ```ignore
//! let mut ds = Dataset::new();
//! ds.label_shapes(&[leaf_shape_a, leaf_shape_b], 1); // class 1 = "leaf"
//! ds.label_shapes(&[branch_shape], 2);                // class 2 = "branch"
//! camera.dispatch(&engine, &model, &state)?;
//! let shape_image = camera.download_shape_index(&engine)?;
//! let frame = ds.annotate(0 /*camera_id*/, shape_image, cfg.width, cfg.height);
//! frame.write_bbox_json(&mut json_out)?;
//! frame.write_class_image_pgm(&mut pgm_out)?;
//! ```
//!
//! Pure-CPU: no GPU resources, no extra dependencies. The PGM writer
//! uses 8-bit ASCII PGM so a test can verify the file shape without
//! pulling in a PNG codec; PNG bindings can be layered on top later.

#![cfg(feature = "gpu")]

use molla_core::{Error, Result};
use std::collections::HashMap;
use std::io::Write;

/// Sentinel shape-id `TiledCamera` writes for a missed pixel
/// (`0xFFFFFFFF`). Pixels carrying this value are treated as
/// "background" everywhere in this module.
pub const MISS_SHAPE_ID: u32 = u32::MAX;

/// 2-D pixel bounding box (axis-aligned), inclusive in both ends.
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox2D {
    /// Tightest inclusive pixel rectangle.
    pub x_min: u32,
    pub y_min: u32,
    pub x_max: u32,
    pub y_max: u32,
    /// Source shape (`Model::shape_geometry` index).
    pub shape_id: u32,
    /// Class assigned by the [`Dataset`]. `0` = unlabelled.
    pub class_id: u32,
    /// Pixel-count for the bbox's "pixel IOU" / area metric.
    pub pixel_count: u32,
}

impl BoundingBox2D {
    /// Width in pixels (inclusive range).
    pub fn width(&self) -> u32 {
        self.x_max.saturating_sub(self.x_min) + 1
    }
    /// Height in pixels (inclusive range).
    pub fn height(&self) -> u32 {
        self.y_max.saturating_sub(self.y_min) + 1
    }
}

/// Single-camera annotation frame: the underlying shape-id image,
/// the derived class image, and the per-instance bounding boxes.
#[derive(Debug, Clone)]
pub struct AnnotationFrame {
    pub camera_id: u32,
    pub width: u32,
    pub height: u32,
    /// Per-pixel shape-id (copied from `TiledCamera::download_shape_index`).
    pub shape_image: Vec<u32>,
    /// Per-pixel class-id (mapped through `Dataset`). Pixels whose
    /// shape isn't labelled get class `0`.
    pub class_image: Vec<u32>,
    /// One bounding box per instance (= per distinct labelled
    /// shape-id with ≥ 1 pixel in this frame).
    pub bboxes: Vec<BoundingBox2D>,
}

impl AnnotationFrame {
    /// Write the bounding-box list as JSON. Each entry has
    /// `{shape_id, class_id, bbox: [x, y, w, h], pixel_count}` —
    /// matching the COCO detection convention `(x, y, w, h)`.
    pub fn write_bbox_json<W: Write>(&self, out: &mut W) -> Result<()> {
        let map_err = |e: std::io::Error| Error::Storage(format!("write_bbox_json: {e}"));
        writeln!(out, "[").map_err(map_err)?;
        for (i, b) in self.bboxes.iter().enumerate() {
            let comma = if i + 1 < self.bboxes.len() { "," } else { "" };
            writeln!(
                out,
                "  {{\"shape_id\": {}, \"class_id\": {}, \"bbox\": [{}, {}, {}, {}], \"pixel_count\": {}}}{comma}",
                b.shape_id,
                b.class_id,
                b.x_min,
                b.y_min,
                b.width(),
                b.height(),
                b.pixel_count
            )
            .map_err(map_err)?;
        }
        writeln!(out, "]").map_err(map_err)?;
        Ok(())
    }

    /// Write the class image as 8-bit ASCII PGM (P2). Class IDs > 255
    /// are clamped — class IDs are intended to be small enumerated
    /// values for the segmentation-mask use case.
    pub fn write_class_image_pgm<W: Write>(&self, out: &mut W) -> Result<()> {
        let map_err = |e: std::io::Error| Error::Storage(format!("write_class_image_pgm: {e}"));
        writeln!(out, "P2").map_err(map_err)?;
        writeln!(out, "{} {}", self.width, self.height).map_err(map_err)?;
        writeln!(out, "255").map_err(map_err)?;
        for y in 0..self.height {
            for x in 0..self.width {
                let v = self.class_image[(y * self.width + x) as usize].min(255);
                write!(out, "{v} ").map_err(map_err)?;
            }
            writeln!(out).map_err(map_err)?;
        }
        Ok(())
    }

    /// Write the per-instance mask image as PGM, using `shape_id`
    /// directly as the pixel value (clamped to 255). Useful for
    /// instance segmentation when there are ≤ 255 instances.
    pub fn write_instance_image_pgm<W: Write>(&self, out: &mut W) -> Result<()> {
        let map_err = |e: std::io::Error| Error::Storage(format!("write_instance_image_pgm: {e}"));
        writeln!(out, "P2").map_err(map_err)?;
        writeln!(out, "{} {}", self.width, self.height).map_err(map_err)?;
        writeln!(out, "255").map_err(map_err)?;
        for y in 0..self.height {
            for x in 0..self.width {
                let s = self.shape_image[(y * self.width + x) as usize];
                let v = if s == MISS_SHAPE_ID {
                    0
                } else {
                    (s + 1).min(255)
                };
                write!(out, "{v} ").map_err(map_err)?;
            }
            writeln!(out).map_err(map_err)?;
        }
        Ok(())
    }
}

/// Dataset builder — collects shape→class labels, then produces
/// annotation frames from rendered shape-id images.
#[derive(Debug, Clone, Default)]
pub struct Dataset {
    shape_to_class: HashMap<u32, u32>,
}

impl Dataset {
    pub fn new() -> Self {
        Self::default()
    }

    /// Assign `class_id` to every shape in `shape_ids`. Repeated
    /// labels overwrite the previous mapping for that shape.
    pub fn label_shapes(&mut self, shape_ids: &[u32], class_id: u32) -> &mut Self {
        for &s in shape_ids {
            self.shape_to_class.insert(s, class_id);
        }
        self
    }

    /// Number of distinct shapes that have a class assigned.
    pub fn labelled_count(&self) -> usize {
        self.shape_to_class.len()
    }

    /// Look up the class for a shape (or `0` when unlabelled).
    pub fn class_for(&self, shape_id: u32) -> u32 {
        self.shape_to_class.get(&shape_id).copied().unwrap_or(0)
    }

    /// Build an [`AnnotationFrame`] from a rendered shape-id image.
    ///
    /// `shape_image` is the buffer returned by
    /// `TiledCamera::download_shape_index`, sliced to the
    /// `width * height` pixels of one camera (the caller indexes
    /// `&buf[cam_id * width * height..(cam_id + 1) * width * height]`).
    /// Infallible — for the fallible variant that rejects malformed
    /// `width × height` mismatches, see [`Self::annotate_checked`].
    pub fn annotate(
        &self,
        camera_id: u32,
        shape_image: &[u32],
        width: u32,
        height: u32,
    ) -> AnnotationFrame {
        debug_assert_eq!(shape_image.len() as u32, width * height);
        let class_image = class_image(shape_image, &self.shape_to_class);
        let bboxes = bounding_boxes(shape_image, width, height, &self.shape_to_class);
        AnnotationFrame {
            camera_id,
            width,
            height,
            shape_image: shape_image.to_vec(),
            class_image,
            bboxes,
        }
    }

    /// Like [`Self::annotate`] but returns `Err` on a `shape_image`
    /// length / `width × height` mismatch. Use this in production
    /// pipelines where the camera buffer slice came from external
    /// configuration.
    pub fn annotate_checked(
        &self,
        camera_id: u32,
        shape_image: &[u32],
        width: u32,
        height: u32,
    ) -> Result<AnnotationFrame, &'static str> {
        if width == 0 || height == 0 {
            return Err("dataset::annotate: width and height must be non-zero");
        }
        let expected = width as usize * height as usize;
        if shape_image.len() != expected {
            return Err("dataset::annotate: shape_image length does not match width × height");
        }
        Ok(self.annotate(camera_id, shape_image, width, height))
    }
}

/// Map a per-pixel shape-id image to a per-pixel class-id image.
pub fn class_image(shape_image: &[u32], shape_to_class: &HashMap<u32, u32>) -> Vec<u32> {
    shape_image
        .iter()
        .map(|s| {
            if *s == MISS_SHAPE_ID {
                0
            } else {
                shape_to_class.get(s).copied().unwrap_or(0)
            }
        })
        .collect()
}

/// Compute per-instance pixel AABB from a per-pixel shape-id image.
/// Returns one `BoundingBox2D` per distinct shape-id that (a) has at
/// least one pixel and (b) carries a label in `shape_to_class`.
/// Background and unlabelled pixels are skipped.
pub fn bounding_boxes(
    shape_image: &[u32],
    width: u32,
    _height: u32,
    shape_to_class: &HashMap<u32, u32>,
) -> Vec<BoundingBox2D> {
    let mut acc: HashMap<u32, (u32, u32, u32, u32, u32)> = HashMap::new();
    for (idx, &s) in shape_image.iter().enumerate() {
        if s == MISS_SHAPE_ID {
            continue;
        }
        // Skip unlabelled shapes — they're background for this
        // annotation pass.
        if !shape_to_class.contains_key(&s) {
            continue;
        }
        let x = (idx as u32) % width;
        let y = (idx as u32) / width;
        let entry = acc.entry(s).or_insert((u32::MAX, u32::MAX, 0, 0, 0));
        entry.0 = entry.0.min(x);
        entry.1 = entry.1.min(y);
        entry.2 = entry.2.max(x);
        entry.3 = entry.3.max(y);
        entry.4 += 1;
    }
    let mut out: Vec<BoundingBox2D> = acc
        .into_iter()
        .map(|(shape_id, (x0, y0, x1, y1, n))| BoundingBox2D {
            x_min: x0,
            y_min: y0,
            x_max: x1,
            y_max: y1,
            shape_id,
            class_id: shape_to_class.get(&shape_id).copied().unwrap_or(0),
            pixel_count: n,
        })
        .collect();
    out.sort_by_key(|b| b.shape_id);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_image_maps_known_shapes_skipping_misses() {
        let mut m = HashMap::new();
        m.insert(0u32, 5u32);
        m.insert(2u32, 9u32);
        let shape = [0u32, MISS_SHAPE_ID, 2, 3];
        let class = class_image(&shape, &m);
        assert_eq!(class, vec![5, 0, 9, 0]);
    }

    #[test]
    fn bbox_finds_correct_pixel_rectangle() {
        // 4×4 image, shape 7 fills a 2×2 block at (1,1)..=(2,2).
        let mut img = vec![MISS_SHAPE_ID; 16];
        for y in 1..=2 {
            for x in 1..=2 {
                img[(y * 4 + x) as usize] = 7;
            }
        }
        let mut m = HashMap::new();
        m.insert(7u32, 1u32);
        let boxes = bounding_boxes(&img, 4, 4, &m);
        assert_eq!(boxes.len(), 1);
        let b = boxes[0];
        assert_eq!((b.x_min, b.y_min, b.x_max, b.y_max), (1, 1, 2, 2));
        assert_eq!(b.pixel_count, 4);
        assert_eq!(b.class_id, 1);
        assert_eq!(b.width(), 2);
        assert_eq!(b.height(), 2);
    }

    #[test]
    fn unlabelled_shapes_are_not_in_bbox_output() {
        let mut img = vec![MISS_SHAPE_ID; 16];
        img[5] = 7;
        img[10] = 8; // 8 is not labelled.
        let mut m = HashMap::new();
        m.insert(7u32, 1u32);
        let boxes = bounding_boxes(&img, 4, 4, &m);
        // Only shape 7 should appear.
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].shape_id, 7);
    }

    #[test]
    fn bboxes_always_have_well_ordered_inclusive_corners() {
        // Property: every emitted bbox satisfies the basic AABB
        // invariants (x_min ≤ x_max, y_min ≤ y_max, pixel_count > 0,
        // width ≥ 1, height ≥ 1). Run against a randomly-spread
        // labelled image so we exercise several shape ids.
        let w = 8;
        let h = 8;
        let mut img = vec![MISS_SHAPE_ID; (w * h) as usize];
        // Place a handful of labelled pixels for shape 7 and shape 9.
        img[3] = 7; // (3, 0)
        img[9] = 7; // (1, 1)
        img[20] = 7; // (4, 2)
        img[33] = 9; // (1, 4)
        img[40] = 9; // (0, 5)
        let mut m = HashMap::new();
        m.insert(7u32, 1u32);
        m.insert(9u32, 2u32);
        let boxes = bounding_boxes(&img, w, h, &m);
        assert!(!boxes.is_empty());
        for b in &boxes {
            assert!(b.x_min <= b.x_max, "{b:?}: x_min > x_max");
            assert!(b.y_min <= b.y_max, "{b:?}: y_min > y_max");
            assert!(b.pixel_count > 0, "{b:?}: empty bbox");
            assert!(b.width() >= 1);
            assert!(b.height() >= 1);
        }
    }

    #[test]
    fn annotate_checked_rejects_length_mismatch() {
        let ds = Dataset::new();
        let result = ds.annotate_checked(0, &[0u32; 5], 4, 4); // 5 ≠ 16
        assert!(result.is_err());
        // Zero-dim inputs are also rejected.
        let result = ds.annotate_checked(0, &[], 0, 4);
        assert!(result.is_err());
    }

    #[test]
    fn annotate_checked_accepts_well_formed_inputs() {
        let ds = Dataset::new();
        let img = vec![MISS_SHAPE_ID; 16];
        let frame = ds.annotate_checked(0, &img, 4, 4).expect("should accept");
        assert_eq!(frame.width, 4);
        assert_eq!(frame.height, 4);
    }

    #[test]
    fn annotation_frame_writes_consistent_files() {
        let mut img = vec![MISS_SHAPE_ID; 16];
        img[5] = 7;
        let mut ds = Dataset::new();
        ds.label_shapes(&[7], 1);
        let frame = ds.annotate(0, &img, 4, 4);
        assert_eq!(frame.bboxes.len(), 1);

        let mut bbox_json = Vec::new();
        frame.write_bbox_json(&mut bbox_json).unwrap();
        let s = std::str::from_utf8(&bbox_json).unwrap();
        assert!(s.contains("\"shape_id\": 7"));
        assert!(s.contains("\"class_id\": 1"));

        let mut pgm = Vec::new();
        frame.write_class_image_pgm(&mut pgm).unwrap();
        assert!(pgm.starts_with(b"P2\n4 4\n255\n"));
    }
}
