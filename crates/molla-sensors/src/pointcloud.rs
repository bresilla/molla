//! H6 — Scan-to-point-cloud conversion + plain-text export.
//!
//! Helios's `HitPoint` ties each return back to the scan grid (row,
//! column, scan id) and carries per-hit scalars; the
//! [`HitPoint`](struct@HitPoint) here is the same shape, plus the
//! incident direction, intensity, and an optional RGB. `Scan::to_hit_points`
//! converts a [`Scan`] into a `Vec<HitPoint>` for one scan id;
//! callers can concatenate clouds from multiple scans before export.
//!
//! Three ASCII writers live here:
//!
//!   * [`write_xyz_rgb`] — generic `x y z r g b` (one hit per line,
//!     misses skipped). The default interchange format.
//!   * [`write_ply_ascii`] — standard PLY ASCII vertex cloud,
//!     consumable by MeshLab / CloudCompare / Open3D.
//!   * [`write_ptx`] — Leica's terrestrial-scanner format (used by
//!     helios's `exportPointCloudPTX`); per-cell output, misses
//!     emitted as `0 0 0 0.5`.
//!
//! All three are pure-CPU and `std::io::Write`-generic so callers can
//! stream to disk, `Vec<u8>`, or an in-memory string. A round-trip
//! `write → read` parser for XYZRGB lives in [`read_xyz_rgb`] for the
//! oracle test.

#![cfg(feature = "gpu")]

use crate::lidar::Scan;
use crate::raycast::RayHit;
use molla_core::{Error, Result};
use molla_math::{Quat, Real, Transform, Vec3};
use std::collections::HashMap;
use std::io::{BufRead, Write};

/// One point-cloud return. Mirrors helios's `HitPoint`.
#[derive(Debug, Clone)]
pub struct HitPoint {
    /// World-space hit position.
    pub position: Vec3,
    /// World-space outward surface normal at the hit.
    pub normal: Vec3,
    /// World-space ray direction that produced this return.
    pub direction: Vec3,
    /// (θ, φ) — zenithal and azimuthal angle of the originating cell.
    pub theta: Real,
    pub phi: Real,
    /// Scan-grid coordinates of the cell that produced this hit.
    pub row: u32,
    pub col: u32,
    /// Sub-ray index within the cell (`0` is the central ray).
    pub subray: u32,
    /// Scan identifier — lets callers tag returns from multiple scans
    /// merged into one cloud.
    pub scan_id: u32,
    /// Scalar intensity (0..1). Default = 1.0 (treat every return as
    /// fully reflective until a material model overrides it).
    pub intensity: f32,
    /// Optional RGB colour (0..1). `None` → an exporter writes white.
    pub color: Option<[f32; 3]>,
    /// Free-form per-hit scalar bag — surface kind, body id, range
    /// rate, anything downstream wants to bind to the cloud.
    pub scalars: HashMap<&'static str, f32>,
}

impl HitPoint {
    /// Convenience: clone-cheap RGB read (white when unset).
    pub fn rgb(&self) -> [f32; 3] {
        self.color.unwrap_or([1.0, 1.0, 1.0])
    }
}

impl Scan {
    /// Convert this scan into a `Vec<HitPoint>` tagged with `scan_id`.
    /// Misses are omitted. The (row, col) and (θ, φ) back-pointers
    /// come from the original metadata.
    pub fn to_hit_points(&self, scan_id: u32) -> Vec<HitPoint> {
        let meta = self.metadata;
        let k = meta.effective_subray_count();
        let mut out = Vec::with_capacity(self.hits.len());
        for row in 0..meta.n_theta {
            let theta = sample_axis(meta.theta_range, meta.n_theta, row);
            for col in 0..meta.n_phi {
                let phi = sample_axis(meta.phi_range, meta.n_phi, col);
                let direction = meta.rc_to_direction(row, col);
                for subray in 0..k {
                    let h = &self.hits[hit_index(self, row, col, subray)];
                    if h.shape_id.is_none() {
                        continue;
                    }
                    out.push(HitPoint {
                        position: h.point,
                        normal: h.normal,
                        direction,
                        theta,
                        phi,
                        row,
                        col,
                        subray,
                        scan_id,
                        intensity: 1.0,
                        color: None,
                        scalars: HashMap::new(),
                    });
                }
            }
        }
        out
    }
}

fn hit_index(scan: &Scan, row: u32, col: u32, subray: u32) -> usize {
    let k = scan.metadata.effective_subray_count();
    ((row * scan.metadata.n_phi + col) * k + subray) as usize
}

fn sample_axis(range: (Real, Real), n: u32, i: u32) -> Real {
    if n <= 1 {
        return range.0;
    }
    range.0 + (range.1 - range.0) * (i as Real / (n - 1) as Real)
}

/// Write `hits` as ASCII `x y z r g b` (one return per line, misses
/// already filtered upstream). Use this when the consumer just wants
/// XYZRGB and doesn't need any header.
pub fn write_xyz_rgb<W: Write>(hits: &[HitPoint], out: &mut W) -> Result<()> {
    for h in hits {
        let rgb = h.rgb();
        writeln!(
            out,
            "{:.6} {:.6} {:.6} {:.4} {:.4} {:.4}",
            h.position.x, h.position.y, h.position.z, rgb[0], rgb[1], rgb[2]
        )
        .map_err(|e| Error::Storage(format!("write_xyz_rgb: {e}")))?;
    }
    Ok(())
}

/// Parse the output of [`write_xyz_rgb`] back into raw `(position, rgb)`
/// pairs. Used by the round-trip oracle test; production code should
/// consume the original `Vec<HitPoint>` (this loses the back-pointer).
pub fn read_xyz_rgb<R: BufRead>(input: R) -> Result<Vec<(Vec3, [f32; 3])>> {
    let mut out = Vec::new();
    for (lineno, line) in input.lines().enumerate() {
        let line = line.map_err(|e| Error::Storage(format!("read_xyz_rgb line {lineno}: {e}")))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_ascii_whitespace().collect();
        if parts.len() < 3 {
            return Err(Error::Storage(format!(
                "read_xyz_rgb line {lineno}: need ≥ 3 cols, got {}",
                parts.len()
            )));
        }
        let parse_f = |i: usize| {
            parts[i]
                .parse::<f64>()
                .map_err(|e| Error::Storage(format!("read_xyz_rgb line {lineno} col {i}: {e}")))
        };
        let parse_f32 = |i: usize| {
            parts[i]
                .parse::<f32>()
                .map_err(|e| Error::Storage(format!("read_xyz_rgb line {lineno} col {i}: {e}")))
        };
        let pos = Vec3::new(parse_f(0)?, parse_f(1)?, parse_f(2)?);
        let rgb = if parts.len() >= 6 {
            [parse_f32(3)?, parse_f32(4)?, parse_f32(5)?]
        } else {
            [1.0, 1.0, 1.0]
        };
        out.push((pos, rgb));
    }
    Ok(out)
}

/// Write `hits` as a PLY ASCII vertex cloud. Standard header +
/// `x y z nx ny nz red green blue` per vertex (RGB scaled 0..255).
pub fn write_ply_ascii<W: Write>(hits: &[HitPoint], out: &mut W) -> Result<()> {
    let map_err = |e: std::io::Error| Error::Storage(format!("write_ply_ascii: {e}"));
    writeln!(out, "ply").map_err(map_err)?;
    writeln!(out, "format ascii 1.0").map_err(map_err)?;
    writeln!(out, "comment molla H6 point-cloud export").map_err(map_err)?;
    writeln!(out, "element vertex {}", hits.len()).map_err(map_err)?;
    for (name, ty) in [
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
        ("nx", "float"),
        ("ny", "float"),
        ("nz", "float"),
        ("red", "uchar"),
        ("green", "uchar"),
        ("blue", "uchar"),
    ] {
        writeln!(out, "property {ty} {name}").map_err(map_err)?;
    }
    writeln!(out, "end_header").map_err(map_err)?;
    for h in hits {
        let rgb = h.rgb();
        let r = (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8;
        writeln!(
            out,
            "{:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {} {} {}",
            h.position.x, h.position.y, h.position.z, h.normal.x, h.normal.y, h.normal.z, r, g, b,
        )
        .map_err(map_err)?;
    }
    Ok(())
}

/// Write a Leica-PTX (terrestrial-scanner) file for one `scan`.
/// Per-cell layout (row-major over θ, *first sub-ray only*); misses
/// emit the conventional `0 0 0 0.5` "no return" sentinel.
pub fn write_ptx<W: Write>(scan: &Scan, out: &mut W) -> Result<()> {
    let meta = scan.metadata;
    let map_err = |e: std::io::Error| Error::Storage(format!("write_ptx: {e}"));
    // Header (helios convention): n_cols n_rows, scanner origin,
    // local-frame axes, and a 4×4 identity transform (the cloud is
    // already in world coordinates because the rays were fired in
    // world space).
    writeln!(out, "{}", meta.n_phi).map_err(map_err)?;
    writeln!(out, "{}", meta.n_theta).map_err(map_err)?;
    writeln!(
        out,
        "{:.6} {:.6} {:.6}",
        meta.origin.x, meta.origin.y, meta.origin.z
    )
    .map_err(map_err)?;
    // Three rotation rows + a translation row → 4×4 identity here.
    writeln!(out, "1.000000 0.000000 0.000000").map_err(map_err)?;
    writeln!(out, "0.000000 1.000000 0.000000").map_err(map_err)?;
    writeln!(out, "0.000000 0.000000 1.000000").map_err(map_err)?;
    writeln!(out, "1.000000 0.000000 0.000000 0.000000").map_err(map_err)?;
    writeln!(out, "0.000000 1.000000 0.000000 0.000000").map_err(map_err)?;
    writeln!(out, "0.000000 0.000000 1.000000 0.000000").map_err(map_err)?;
    writeln!(out, "0.000000 0.000000 0.000000 1.000000").map_err(map_err)?;
    for row in 0..meta.n_theta {
        for col in 0..meta.n_phi {
            match scan.first_hit_at(row, col) {
                Some(h) => writeln!(
                    out,
                    "{:.6} {:.6} {:.6} {:.4}",
                    h.point.x, h.point.y, h.point.z, 1.0_f32
                )
                .map_err(map_err)?,
                None => writeln!(out, "0 0 0 0.5").map_err(map_err)?,
            }
        }
    }
    Ok(())
}

// ── H8 — Cloud filters, voxel binning, scan registration ────────────

/// Numeric comparator for [`scalar_filter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compare {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
}

impl Compare {
    fn test(self, a: f32, b: f32) -> bool {
        match self {
            Compare::Lt => a < b,
            Compare::Le => a <= b,
            Compare::Gt => a > b,
            Compare::Ge => a >= b,
            Compare::Eq => (a - b).abs() < 1.0e-6,
        }
    }
}

/// Remove points farther than `max_distance` (m) from `origin`.
pub fn distance_filter(points: &[HitPoint], origin: Vec3, max_distance: Real) -> Vec<HitPoint> {
    let max2 = max_distance * max_distance;
    points
        .iter()
        .filter(|h| {
            let d = h.position - origin;
            d.x * d.x + d.y * d.y + d.z * d.z <= max2
        })
        .cloned()
        .collect()
}

/// Keep or drop points inside the given AABB. `keep_inside = true`
/// retains points within `[min, max]` (inclusive); `false` is the
/// "cut out a box" operation.
pub fn xyz_filter(points: &[HitPoint], min: Vec3, max: Vec3, keep_inside: bool) -> Vec<HitPoint> {
    let inside = |p: Vec3| {
        p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y && p.z >= min.z && p.z <= max.z
    };
    points
        .iter()
        .filter(|h| inside(h.position) == keep_inside)
        .cloned()
        .collect()
}

/// Drop points whose `intensity` is below `min_intensity`.
pub fn reflectance_filter(points: &[HitPoint], min_intensity: f32) -> Vec<HitPoint> {
    points
        .iter()
        .filter(|h| h.intensity >= min_intensity)
        .cloned()
        .collect()
}

/// Filter on a per-hit scalar (a `scalars` bag key). Hits without the
/// key are dropped.
pub fn scalar_filter(
    points: &[HitPoint],
    key: &'static str,
    threshold: f32,
    cmp: Compare,
) -> Vec<HitPoint> {
    points
        .iter()
        .filter(|h| {
            h.scalars
                .get(key)
                .copied()
                .is_some_and(|v| cmp.test(v, threshold))
        })
        .cloned()
        .collect()
}

/// Apply a rigid transform to a cloud in place (position + normal +
/// direction). Use [`translate_cloud`] / [`rotate_cloud`] for the
/// single-axis convenience wrappers.
pub fn transform_cloud(points: &mut [HitPoint], xf: Transform) {
    for h in points.iter_mut() {
        h.position = xf.transform_point(h.position);
        h.normal = xf.transform_vector(h.normal);
        h.direction = xf.transform_vector(h.direction);
    }
}

/// Translate every point in the cloud by `shift`. Equivalent to a
/// pure-translation `transform_cloud`.
pub fn translate_cloud(points: &mut [HitPoint], shift: Vec3) {
    transform_cloud(points, Transform::from_translation(shift));
}

/// Rotate every point in the cloud by `rotation` around the origin.
pub fn rotate_cloud(points: &mut [HitPoint], rotation: Quat) {
    transform_cloud(points, Transform::from_rotation(rotation));
}

/// One bin of a regular-grid binning. The grid divides the AABB
/// `[grid_min, grid_max]` into `divisions = (nx, ny, nz)` cells.
#[derive(Debug, Clone)]
pub struct GridCell {
    /// Integer cell index in (i, j, k) along (x, y, z).
    pub ijk: (u32, u32, u32),
    /// World-space cell centre.
    pub center: Vec3,
    /// World-space cell size (`(grid_max - grid_min) / divisions`).
    pub size: Vec3,
    /// Indices (into the input cloud) of every point inside this cell.
    pub point_indices: Vec<u32>,
}

impl GridCell {
    pub fn point_count(&self) -> u32 {
        self.point_indices.len() as u32
    }
}

/// Partition a cloud into a regular grid. Returns one [`GridCell`]
/// per bin in row-major `(i, j, k)` order with `j` major outermost
/// (`k * (nx * ny) + j * nx + i`). Empty cells are included so the
/// caller can index by `(i, j, k)` directly.
pub fn bin_into_grid(
    points: &[HitPoint],
    grid_min: Vec3,
    grid_max: Vec3,
    divisions: (u32, u32, u32),
) -> Vec<GridCell> {
    let (nx, ny, nz) = divisions;
    let nx = nx.max(1);
    let ny = ny.max(1);
    let nz = nz.max(1);
    // Sort the AABB so we tolerate callers passing `(max, min)` by
    // mistake — the cell-size logic below assumes `max ≥ min` per
    // axis. The cells still come out in the same indexing order.
    let lo = Vec3::new(
        grid_min.x.min(grid_max.x),
        grid_min.y.min(grid_max.y),
        grid_min.z.min(grid_max.z),
    );
    let hi = Vec3::new(
        grid_min.x.max(grid_max.x),
        grid_min.y.max(grid_max.y),
        grid_min.z.max(grid_max.z),
    );
    let grid_min = lo;
    let grid_max = hi;
    let size = Vec3::new(
        (grid_max.x - grid_min.x) / nx as Real,
        (grid_max.y - grid_min.y) / ny as Real,
        (grid_max.z - grid_min.z) / nz as Real,
    );
    let total = (nx * ny * nz) as usize;
    let mut cells: Vec<GridCell> = (0..total)
        .map(|idx| {
            let idx = idx as u32;
            let i = idx % nx;
            let j = (idx / nx) % ny;
            let k = idx / (nx * ny);
            let center = Vec3::new(
                grid_min.x + (i as Real + 0.5) * size.x,
                grid_min.y + (j as Real + 0.5) * size.y,
                grid_min.z + (k as Real + 0.5) * size.z,
            );
            GridCell {
                ijk: (i, j, k),
                center,
                size,
                point_indices: Vec::new(),
            }
        })
        .collect();
    for (pi, h) in points.iter().enumerate() {
        let fx = (h.position.x - grid_min.x) / size.x.max(1.0e-12);
        let fy = (h.position.y - grid_min.y) / size.y.max(1.0e-12);
        let fz = (h.position.z - grid_min.z) / size.z.max(1.0e-12);
        if fx < 0.0 || fy < 0.0 || fz < 0.0 {
            continue;
        }
        let i = fx.floor() as i64;
        let j = fy.floor() as i64;
        let k = fz.floor() as i64;
        if i < 0 || j < 0 || k < 0 || i >= nx as i64 || j >= ny as i64 || k >= nz as i64 {
            continue;
        }
        let idx = (k as u32) * (nx * ny) + (j as u32) * nx + (i as u32);
        cells[idx as usize].point_indices.push(pi as u32);
    }
    cells
}

/// Helper: convert a `RayHit` to the same miss sentinel `Scan::hit_at`
/// emits. Useful when downstream code wants a flat `[RayHit; cells]`
/// view of the first-hit channel.
pub fn first_hit_or_miss(hit: Option<&RayHit>) -> RayHit {
    match hit {
        Some(h) => *h,
        None => RayHit {
            shape_id: None,
            body_id: molla_core::BodyId::NONE,
            t: Real::INFINITY,
            point: Vec3::ZERO,
            normal: Vec3::Y,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lidar::ScanMetadata;

    fn make_scan() -> Scan {
        let mut meta = ScanMetadata::new(Vec3::new(0.0, 5.0, 0.0), 2, 3);
        meta.theta_range = (0.0, std::f64::consts::PI);
        meta.phi_range = (0.0, 2.0 * std::f64::consts::PI);
        // Manually crafted hits: every cell is a hit.
        let mut hits = Vec::new();
        for row in 0..meta.n_theta {
            for col in 0..meta.n_phi {
                hits.push(RayHit {
                    shape_id: Some(row * meta.n_phi + col),
                    body_id: molla_core::BodyId::NONE,
                    t: 1.0 + row as Real + 0.1 * col as Real,
                    point: Vec3::new(row as Real, col as Real, 0.0),
                    normal: Vec3::Y,
                });
            }
        }
        Scan {
            metadata: meta,
            hits,
        }
    }

    #[test]
    fn scan_to_hit_points_emits_one_per_real_hit() {
        let scan = make_scan();
        let pts = scan.to_hit_points(7);
        assert_eq!(pts.len(), scan.hits.len());
        // Sample one and check the back-pointers line up.
        let p = &pts[3];
        assert_eq!(p.scan_id, 7);
        assert_eq!(p.row * scan.metadata.n_phi + p.col, 3);
    }

    #[test]
    fn xyz_rgb_round_trips_through_ascii() {
        let scan = make_scan();
        let pts = scan.to_hit_points(0);
        let mut buf: Vec<u8> = Vec::new();
        write_xyz_rgb(&pts, &mut buf).unwrap();
        let read = read_xyz_rgb(buf.as_slice()).unwrap();
        assert_eq!(read.len(), pts.len());
        for (p, (pos, rgb)) in pts.iter().zip(read.iter()) {
            assert!((p.position - *pos).length() < 1.0e-5);
            for (k, &c) in rgb.iter().enumerate() {
                assert!((p.rgb()[k] - c).abs() < 1.0e-3);
            }
        }
    }

    #[test]
    fn ply_ascii_header_and_vertex_count_are_correct() {
        let scan = make_scan();
        let pts = scan.to_hit_points(0);
        let mut buf: Vec<u8> = Vec::new();
        write_ply_ascii(&pts, &mut buf).unwrap();
        let text = std::str::from_utf8(&buf).unwrap();
        assert!(text.starts_with("ply\n"));
        assert!(text.contains(&format!("element vertex {}", pts.len())));
        // Header has `end_header` exactly once before the body.
        assert_eq!(text.matches("end_header").count(), 1);
    }

    fn fake_point(pos: Vec3, intensity: f32) -> HitPoint {
        HitPoint {
            position: pos,
            normal: Vec3::Y,
            direction: -Vec3::Y,
            theta: 0.0,
            phi: 0.0,
            row: 0,
            col: 0,
            subray: 0,
            scan_id: 0,
            intensity,
            color: None,
            scalars: HashMap::new(),
        }
    }

    #[test]
    fn distance_filter_keeps_points_within_radius() {
        let pts = vec![
            fake_point(Vec3::new(0.0, 0.0, 0.0), 1.0),
            fake_point(Vec3::new(1.0, 0.0, 0.0), 1.0),
            fake_point(Vec3::new(5.0, 0.0, 0.0), 1.0),
        ];
        let kept = distance_filter(&pts, Vec3::ZERO, 2.0);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn xyz_filter_inside_and_outside_modes() {
        let pts = vec![
            fake_point(Vec3::new(0.5, 0.5, 0.5), 1.0),
            fake_point(Vec3::new(5.0, 5.0, 5.0), 1.0),
        ];
        let inside = xyz_filter(&pts, Vec3::ZERO, Vec3::splat(1.0), true);
        let outside = xyz_filter(&pts, Vec3::ZERO, Vec3::splat(1.0), false);
        assert_eq!(inside.len(), 1);
        assert_eq!(outside.len(), 1);
        assert!((inside[0].position - Vec3::splat(0.5)).length() < 1.0e-6);
        assert!((outside[0].position - Vec3::splat(5.0)).length() < 1.0e-6);
    }

    #[test]
    fn reflectance_filter_drops_low_intensity() {
        let pts = vec![
            fake_point(Vec3::ZERO, 0.2),
            fake_point(Vec3::X, 0.6),
            fake_point(Vec3::Y, 0.9),
        ];
        let kept = reflectance_filter(&pts, 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn scalar_filter_uses_compare_predicate() {
        let mut pts = Vec::new();
        for (i, v) in [0.1f32, 0.5, 0.9].iter().enumerate() {
            let mut p = fake_point(Vec3::new(i as Real, 0.0, 0.0), 1.0);
            p.scalars.insert("range_rate", *v);
            pts.push(p);
        }
        let lt = scalar_filter(&pts, "range_rate", 0.5, Compare::Lt);
        let ge = scalar_filter(&pts, "range_rate", 0.5, Compare::Ge);
        assert_eq!(lt.len(), 1);
        assert_eq!(ge.len(), 2);
    }

    #[test]
    fn transform_cloud_shifts_positions_and_rotates_vectors() {
        let mut pts = vec![fake_point(Vec3::new(1.0, 0.0, 0.0), 1.0)];
        translate_cloud(&mut pts, Vec3::new(0.0, 2.0, 0.0));
        assert!((pts[0].position - Vec3::new(1.0, 2.0, 0.0)).length() < 1.0e-9);
        // 90° rotation about +Y maps +X → +Z and rotates the normal.
        let rot = Quat::from_axis_angle(Vec3::Y, std::f64::consts::FRAC_PI_2);
        rotate_cloud(&mut pts, rot);
        assert!((pts[0].normal - Vec3::Y).length() < 1.0e-9);
    }

    #[test]
    fn read_xyz_rgb_tolerates_comments_blank_lines_and_crlf() {
        // Mixed comments, blanks, CRLF, trailing whitespace.
        let input = "# header comment\r\n\
                     1.0 2.0 3.0\r\n\
                     \r\n\
                     # mid comment\n\
                     4.0 5.0 6.0 0.5 0.6 0.7  \n\
                     \n";
        let read = read_xyz_rgb(input.as_bytes()).unwrap();
        assert_eq!(read.len(), 2);
        assert!((read[0].0 - Vec3::new(1.0, 2.0, 3.0)).length() < 1.0e-9);
        // No RGB on the first line → default white.
        assert_eq!(read[0].1, [1.0, 1.0, 1.0]);
        assert!((read[1].0 - Vec3::new(4.0, 5.0, 6.0)).length() < 1.0e-9);
        assert!((read[1].1[0] - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn read_xyz_rgb_rejects_short_lines() {
        // Only 2 columns — must error.
        let input = "1.0 2.0\n";
        assert!(read_xyz_rgb(input.as_bytes()).is_err());
    }

    #[test]
    fn read_xyz_rgb_rejects_non_numeric() {
        let input = "1.0 foo 3.0\n";
        assert!(read_xyz_rgb(input.as_bytes()).is_err());
    }

    #[test]
    fn empty_cloud_filters_return_empty_results() {
        let empty: Vec<HitPoint> = Vec::new();
        assert!(distance_filter(&empty, Vec3::ZERO, 5.0).is_empty());
        assert!(xyz_filter(&empty, Vec3::ZERO, Vec3::splat(1.0), true).is_empty());
        assert!(reflectance_filter(&empty, 0.5).is_empty());
        assert!(scalar_filter(&empty, "k", 0.0, Compare::Lt).is_empty());
    }

    #[test]
    fn bin_into_grid_returns_full_cell_grid_for_empty_cloud() {
        let cells = bin_into_grid(&[], Vec3::ZERO, Vec3::splat(1.0), (2, 2, 2));
        assert_eq!(cells.len(), 8);
        assert!(cells.iter().all(|c| c.point_count() == 0));
    }

    #[test]
    fn bin_into_grid_tolerates_inverted_aabb() {
        // Pass `(max, min)` instead of `(min, max)`; the helper
        // should sort internally and bin correctly.
        let pts = vec![fake_point(Vec3::new(0.5, 0.5, 0.5), 1.0)];
        let cells = bin_into_grid(&pts, Vec3::splat(1.0), Vec3::ZERO, (2, 2, 2));
        let total: u32 = cells.iter().map(|c| c.point_count()).sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn bin_into_grid_clamps_zero_divisions_up() {
        // (0, 0, 0) → treated as (1, 1, 1) to avoid div-by-zero.
        let pts = vec![fake_point(Vec3::new(0.5, 0.5, 0.5), 1.0)];
        let cells = bin_into_grid(&pts, Vec3::ZERO, Vec3::splat(1.0), (0, 0, 0));
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].point_count(), 1);
    }

    #[test]
    fn bin_into_grid_groups_points_into_correct_cells() {
        let pts = vec![
            fake_point(Vec3::new(0.1, 0.1, 0.1), 1.0),
            fake_point(Vec3::new(0.6, 0.6, 0.6), 1.0),
            fake_point(Vec3::new(5.0, 5.0, 5.0), 1.0), // out of grid
        ];
        let cells = bin_into_grid(&pts, Vec3::ZERO, Vec3::splat(1.0), (2, 2, 2));
        assert_eq!(cells.len(), 8);
        let total: u32 = cells.iter().map(|c| c.point_count()).sum();
        assert_eq!(total, 2, "expected 2 in-grid points binned, 1 dropped");
        // Cell at (0,0,0) holds the first point.
        let c000 = cells.iter().find(|c| c.ijk == (0, 0, 0)).unwrap();
        assert_eq!(c000.point_count(), 1);
        // Cell at (1,1,1) holds the second.
        let c111 = cells.iter().find(|c| c.ijk == (1, 1, 1)).unwrap();
        assert_eq!(c111.point_count(), 1);
    }

    #[test]
    fn ptx_emits_one_line_per_cell_with_miss_sentinels() {
        let mut scan = make_scan();
        // Mark one cell as a miss; PTX must emit `0 0 0 0.5` there.
        scan.hits[1].shape_id = None;
        let mut buf: Vec<u8> = Vec::new();
        write_ptx(&scan, &mut buf).unwrap();
        let text = std::str::from_utf8(&buf).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        // Helios PTX: 2 axis-counts + origin + 3 rotation rows + 4
        // transform rows = 10 header lines.
        let header_lines = 10;
        let body = &lines[header_lines..];
        assert_eq!(
            body.len(),
            (scan.metadata.n_theta * scan.metadata.n_phi) as usize,
            "expected one line per cell"
        );
        assert!(
            body.iter().any(|l| l.trim() == "0 0 0 0.5"),
            "expected at least one miss sentinel"
        );
    }
}
