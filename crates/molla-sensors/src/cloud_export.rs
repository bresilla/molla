//! CPU serialization of explicitly collected packed point ranges.

use crate::CloudPoint;
use molla_core::{Error, Result};
use std::io::{BufRead, Read, Write};

const PREFIX: [&str; 3] = [
    "ply",
    "format binary_little_endian 1.0",
    "comment molla packed cloud v1",
];
const PROPERTIES: [&str; 31] = [
    "float x",
    "float y",
    "float z",
    "float intensity",
    "float nx",
    "float ny",
    "float nz",
    "float range",
    "float dx",
    "float dy",
    "float dz",
    "float scalar",
    "uint sensor_id_low",
    "uint sensor_id_high",
    "uint sample_id_low",
    "uint sample_id_high",
    "uint row",
    "uint column",
    "uint subray",
    "uint source_index",
    "uint shape_id",
    "uchar red",
    "uchar green",
    "uchar blue",
    "uchar alpha",
    "uint flags",
    "uint reserved0",
    "float theta",
    "float phi",
    "float reserved1",
    "float reserved2",
];
const RECORD_BYTES: usize = 112;
const LINE_LIMIT: u64 = 512;

/// Limits apply to the declared point allocation and this frame's header plus payload bytes.
#[derive(Debug, Clone, Copy)]
pub struct CloudReadLimits {
    pub max_points: usize,
    pub max_file_bytes: u64,
}

impl Default for CloudReadLimits {
    fn default() -> Self {
        Self {
            max_points: 1_000_000,
            max_file_bytes: 128 * 1024 * 1024,
        }
    }
}

/// Writes the complete 112-byte record as versioned binary little-endian PLY properties.
pub fn write_cloud_ply<W: Write>(points: &[CloudPoint], out: &mut W) -> Result<()> {
    let io = |e| Error::Storage(format!("write_cloud_ply: {e}"));
    for line in PREFIX {
        writeln!(out, "{line}").map_err(io)?;
    }
    writeln!(out, "element vertex {}", points.len()).map_err(io)?;
    for property in PROPERTIES {
        writeln!(out, "property {property}").map_err(io)?;
    }
    writeln!(out, "end_header").map_err(io)?;
    for point in points {
        let words: [u32; 28] = bytemuck::cast(*point);
        let mut bytes = [0u8; RECORD_BYTES];
        for (chunk, word) in bytes.chunks_exact_mut(4).zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        out.write_all(&bytes).map_err(io)?;
    }
    Ok(())
}

/// Reads one packed-cloud v1 PLY frame, leaving subsequent stream bytes untouched.
pub fn read_cloud_ply<R: BufRead>(
    mut input: R,
    limits: CloudReadLimits,
) -> Result<Vec<CloudPoint>> {
    let mut header_bytes = 0u64;
    for expected in PREFIX {
        if line(&mut input, &mut header_bytes, limits.max_file_bytes)? != expected {
            return Err(Error::Storage(
                "read_cloud_ply: unsupported format or schema version".into(),
            ));
        }
    }
    let vertex_line = line(&mut input, &mut header_bytes, limits.max_file_bytes)?;
    let count = vertex_line
        .strip_prefix("element vertex ")
        .filter(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|n| n.parse::<usize>().ok())
        .filter(|&n| n <= limits.max_points)
        .ok_or_else(|| {
            Error::Storage("read_cloud_ply: invalid or over-budget vertex count".into())
        })?;
    let payload = (count as u64)
        .checked_mul(RECORD_BYTES as u64)
        .ok_or_else(|| Error::Storage("read_cloud_ply: payload size overflow".into()))?;
    for property in PROPERTIES {
        if line(&mut input, &mut header_bytes, limits.max_file_bytes)?
            != format!("property {property}")
        {
            return Err(Error::Storage(
                "read_cloud_ply: unsupported property layout".into(),
            ));
        }
    }
    if line(&mut input, &mut header_bytes, limits.max_file_bytes)? != "end_header"
        || header_bytes
            .checked_add(payload)
            .is_none_or(|n| n > limits.max_file_bytes)
    {
        return Err(Error::Storage(
            "read_cloud_ply: missing header terminator or file budget exceeded".into(),
        ));
    }
    let mut points = Vec::new();
    points
        .try_reserve_exact(count)
        .map_err(|_| Error::Storage("read_cloud_ply: point allocation failed".into()))?;
    for _ in 0..count {
        let mut bytes = [0u8; RECORD_BYTES];
        input
            .read_exact(&mut bytes)
            .map_err(|e| Error::Storage(format!("read_cloud_ply: {e}")))?;
        let words = std::array::from_fn::<_, 28, _>(|i| {
            u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
        });
        points.push(bytemuck::cast(words));
    }
    Ok(points)
}

fn line<R: BufRead>(input: &mut R, bytes: &mut u64, budget: u64) -> Result<String> {
    let mut line = Vec::new();
    (&mut *input)
        .take(LINE_LIMIT + 1)
        .read_until(b'\n', &mut line)
        .map_err(|e| Error::Storage(format!("read_cloud_ply: {e}")))?;
    *bytes = bytes
        .checked_add(line.len() as u64)
        .ok_or_else(|| Error::Storage("read_cloud_ply: header overflow".into()))?;
    if line.len() as u64 > LINE_LIMIT || *bytes > budget || line.last() != Some(&b'\n') {
        return Err(Error::Storage(
            "read_cloud_ply: truncated or over-budget header line".into(),
        ));
    }
    line.pop();
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    String::from_utf8(line).map_err(|_| Error::Storage("read_cloud_ply: non-UTF8 header".into()))
}

/// Writes the lossy legacy XYZRGB projection; requires valid flags and finite XYZ for every selected point.
pub fn write_cloud_xyz_rgb<W: Write>(points: &[CloudPoint], out: &mut W) -> Result<()> {
    if points
        .iter()
        .any(|p| p.extra[2] & 1 == 0 || p.position[..3].iter().any(|v| !v.is_finite()))
    {
        return Err(Error::Storage(
            "write_cloud_xyz_rgb: invalid or missing selected point".into(),
        ));
    }
    for point in points {
        let rgb = if point.extra[2] & 2 == 0 {
            [1.; 3]
        } else {
            std::array::from_fn(|i| ((point.extra[1] >> (i * 8)) & 255) as f32 / 255.)
        };
        writeln!(
            out,
            "{:.6} {:.6} {:.6} {:.4} {:.4} {:.4}",
            point.position[0], point.position[1], point.position[2], rgb[0], rgb[1], rgb[2]
        )
        .map_err(|e| Error::Storage(format!("write_cloud_xyz_rgb: {e}")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    fn arbitrary_points() -> Vec<CloudPoint> {
        (0..31)
            .map(|i| {
                let words: [u32; 28] = std::array::from_fn(|j| {
                    [
                        0, 0x80000000, 0x7fc01234, 0x7f800000, 0xff800000, 0xffffffff, 1,
                        0x7f7fffff,
                    ][(i + j) % 8]
                });
                bytemuck::cast(words)
            })
            .collect()
    }

    #[test]
    fn packed_ply_roundtrip_preserves_all_bits_and_stream_boundaries() {
        for points in [vec![], arbitrary_points()] {
            let mut file = Vec::new();
            write_cloud_ply(&points, &mut file).unwrap();
            let limits = CloudReadLimits {
                max_points: points.len(),
                max_file_bytes: file.len() as u64,
            };
            let mut cursor = Cursor::new(file.clone());
            cursor.get_mut().extend_from_slice(b"next frame");
            let decoded = read_cloud_ply(&mut cursor, limits).unwrap();
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(&points),
                bytemuck::cast_slice::<_, u8>(&decoded)
            );
            assert_eq!(cursor.position(), file.len() as u64);
            assert!(
                read_cloud_ply(
                    Cursor::new(&file),
                    CloudReadLimits {
                        max_file_bytes: limits.max_file_bytes - 1,
                        ..limits
                    }
                )
                .is_err()
            );
            if !points.is_empty() {
                assert!(
                    read_cloud_ply(
                        Cursor::new(&file),
                        CloudReadLimits {
                            max_points: points.len() - 1,
                            ..limits
                        }
                    )
                    .is_err()
                );
            }
            let buffered =
                read_cloud_ply(BufReader::with_capacity(7, Cursor::new(file)), limits).unwrap();
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(&points),
                bytemuck::cast_slice::<_, u8>(&buffered)
            );
        }
    }

    #[test]
    fn packed_ply_layout_is_explicit_little_endian_and_accepts_crlf() {
        let point = CloudPoint {
            position: [1., -2., 0.5, 0.25],
            ids: [0x12345678, 0xfedcba98, 7, 9],
            extra: [17, 0x80402010, 3, 0],
            ..Default::default()
        };
        let mut file = Vec::new();
        write_cloud_ply(&[point], &mut file).unwrap();
        let offset = file.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
        let header = std::str::from_utf8(&file[..offset]).unwrap();
        assert_eq!(
            header
                .lines()
                .filter(|line| line.starts_with("property "))
                .count(),
            31
        );
        assert_eq!(file.len() - offset, 112);
        assert_eq!(&file[offset..offset + 4], &[0, 0, 128, 63]);
        assert_eq!(&file[offset + 48..offset + 52], &[0x78, 0x56, 0x34, 0x12]);
        assert_eq!(&file[offset + 84..offset + 88], &[0x10, 0x20, 0x40, 0x80]);
        let mut crlf = header.replace('\n', "\r\n").into_bytes();
        crlf.extend_from_slice(&file[offset..]);
        assert_eq!(
            read_cloud_ply(Cursor::new(crlf), CloudReadLimits::default()).unwrap(),
            vec![point]
        );
    }

    #[test]
    fn packed_ply_rejects_truncation_bad_schema_and_unbounded_headers() {
        let mut file = Vec::new();
        write_cloud_ply(&arbitrary_points()[..1], &mut file).unwrap();
        for n in 0..file.len() {
            assert!(
                read_cloud_ply(Cursor::new(&file[..n]), CloudReadLimits::default()).is_err(),
                "truncated at {n}"
            );
        }
        for (from, to) in [
            ("binary_little_endian", "ascii"),
            ("cloud v1", "cloud v2"),
            ("float x", "double x"),
            ("vertex 1", "vertex 9999999999999999999999999"),
            ("vertex 1", "vertex -1"),
        ] {
            let offset = file.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
            let mut changed = std::str::from_utf8(&file[..offset])
                .unwrap()
                .replace(from, to)
                .into_bytes();
            changed.extend_from_slice(&file[offset..]);
            assert!(read_cloud_ply(Cursor::new(changed), CloudReadLimits::default()).is_err());
        }
        let mut huge = Cursor::new(vec![b'a'; 10_000]);
        assert!(read_cloud_ply(&mut huge, CloudReadLimits::default()).is_err());
        assert_eq!(huge.position(), LINE_LIMIT + 1);
        assert!(read_cloud_ply(Cursor::new(b"\xff\n"), CloudReadLimits::default()).is_err());
    }

    #[test]
    fn xyz_projection_matches_legacy_reader_and_rejects_invalid_points_before_writing() {
        let point = CloudPoint {
            position: [1.25, -2., 0.5, 1.],
            extra: [1, 0x80604020, 3, 0],
            ..Default::default()
        };
        let mut no_color = point;
        no_color.extra[2] = 1;
        let mut file = Vec::new();
        write_cloud_xyz_rgb(&[point, no_color], &mut file).unwrap();
        let read = crate::pointcloud::read_xyz_rgb(Cursor::new(file)).unwrap();
        assert_eq!(read[0].0, molla_math::Vec3::new(1.25, -2., 0.5));
        for (actual, byte) in read[0].1.into_iter().zip([32., 64., 96.]) {
            assert!((actual - byte / 255.).abs() < 0.000051);
        }
        assert_eq!(read[1].1, [1.; 3]);
        for mut bad in [point, CloudPoint::default()] {
            if bad.extra[2] & 1 != 0 {
                bad.position[0] = f32::NAN;
            }
            let mut out = Vec::new();
            assert!(write_cloud_xyz_rgb(&[point, bad], &mut out).is_err());
            assert!(out.is_empty());
        }
    }

    #[test]
    fn cloud_serialization_propagates_stream_errors() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("fixture error"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        assert!(write_cloud_ply(&[], &mut Broken).is_err());
        assert!(
            write_cloud_xyz_rgb(
                &[CloudPoint::new(molla_math::Vec3::ZERO, 0, 0)],
                &mut Broken
            )
            .is_err()
        );
        struct BadRead;
        impl Read for BadRead {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("fixture error"))
            }
        }
        assert!(read_cloud_ply(BufReader::new(BadRead), CloudReadLimits::default()).is_err());
        struct PartialWriter {
            remaining: usize,
        }
        impl Write for PartialWriter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.remaining == 0 {
                    return Err(std::io::Error::other("mid-record write failure"));
                }
                let n = self.remaining.min(bytes.len());
                self.remaining -= n;
                Ok(n)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut header = Vec::new();
        write_cloud_ply(&[], &mut header).unwrap();
        assert!(
            write_cloud_ply(
                &arbitrary_points()[..1],
                &mut PartialWriter {
                    remaining: header.len() + 17
                }
            )
            .is_err()
        );
        struct PartialReader {
            bytes: Cursor<Vec<u8>>,
            remaining: usize,
        }
        impl Read for PartialReader {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                if self.remaining == 0 {
                    return Err(std::io::Error::other("mid-record read failure"));
                }
                let n = out.len().min(self.remaining);
                let read = self.bytes.read(&mut out[..n])?;
                self.remaining -= read;
                Ok(read)
            }
        }
        let mut file = Vec::new();
        write_cloud_ply(&arbitrary_points()[..1], &mut file).unwrap();
        assert!(
            read_cloud_ply(
                BufReader::new(PartialReader {
                    bytes: Cursor::new(file),
                    remaining: header.len() + 17
                }),
                CloudReadLimits::default()
            )
            .is_err()
        );
    }
}
