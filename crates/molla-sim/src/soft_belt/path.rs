use molla_core::{Error, Result};
use molla_math::{Real, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct BeltPulley {
    pub center: Vec3,
    pub radius: Real,
}

#[derive(Clone, Copy, Debug)]
pub struct BeltPathSample {
    pub position: Vec3,
    pub tangent: Vec3,
    pub outward: Vec3,
}

#[derive(Clone, Debug)]
enum Piece {
    Arc {
        pulley: BeltPulley,
        start: Real,
        sweep: Real,
    },
    Line {
        start: Vec3,
        end: Vec3,
        outward: Vec3,
    },
}

impl Piece {
    fn length(&self) -> Real {
        match self {
            Self::Arc { pulley, sweep, .. } => pulley.radius * sweep,
            Self::Line { start, end, .. } => (*end - *start).length(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BeltPath {
    pieces: Vec<Piece>,
    length: Real,
}

impl BeltPath {
    /// Construct a convex CCW YZ polygon with a nonnegative inward miter offset.
    pub fn from_closed_polyline(points: &[Vec3], inset: Real) -> Result<Self> {
        let invalid = || Error::Build("invalid convex belt polyline or inset".into());
        if points.len() < 3
            || points.len() > 4096
            || !inset.is_finite()
            || inset < 0.0
            || points.iter().any(|p| !p.is_finite() || p.x != 0.0)
        {
            return Err(invalid());
        }
        let normals = |points: &[Vec3]| -> Result<Vec<Vec3>> {
            let mut result = Vec::with_capacity(points.len());
            let mut area = 0.0;
            for i in 0..points.len() {
                let edge = points[(i + 1) % points.len()] - points[i];
                let length = edge.length();
                if !length.is_finite() || length <= 1e-12 {
                    return Err(invalid());
                }
                let normal = Vec3::new(0.0, edge.z, -edge.y) / length;
                if points
                    .iter()
                    .any(|p| (*p - points[i]).dot(normal) > 1e-10 * length)
                {
                    return Err(invalid());
                }
                area += (points[i] - points[0]).cross(edge).x;
                result.push(normal);
            }
            if !area.is_finite() || area <= 0.0 {
                return Err(invalid());
            }
            Ok(result)
        };
        let original = normals(points)?;
        let shifted = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let a = original[(i + points.len() - 1) % points.len()];
                let b = original[i];
                *p - (a + b) * (inset / (1.0 + a.dot(b)))
            })
            .collect::<Vec<_>>();
        if shifted.iter().any(|p| !p.is_finite()) {
            return Err(invalid());
        }
        let directions = normals(&shifted)?;
        let mut pieces = Vec::with_capacity(points.len());
        for i in 0..points.len() {
            if directions[i].dot(original[i]) < 1.0 - 1e-10 {
                return Err(invalid());
            }
            pieces.push(Piece::Line {
                start: shifted[i],
                end: shifted[(i + 1) % points.len()],
                outward: directions[i],
            });
        }
        let length = pieces.iter().map(Piece::length).sum();
        Ok(Self { pieces, length })
    }

    /// Construct the outer CCW common-tangent envelope of disjoint pulleys in the YZ plane.
    pub fn around_pulleys(pulleys: &[BeltPulley]) -> Result<Self> {
        let invalid = || Error::Build("invalid or non-convex ordered belt pulleys".into());
        if pulleys.len() < 2
            || pulleys.iter().any(|p| {
                !p.center.is_finite()
                    || p.center.x != 0.0
                    || !p.radius.is_finite()
                    || p.radius <= 0.0
            })
        {
            return Err(invalid());
        }
        for (i, p) in pulleys.iter().enumerate() {
            for q in &pulleys[i + 1..] {
                let distance = (q.center - p.center).length();
                if !distance.is_finite() || distance <= p.radius + q.radius {
                    return Err(invalid());
                }
            }
        }
        let mut normals = Vec::with_capacity(pulleys.len());
        for i in 0..pulleys.len() {
            let a = pulleys[i];
            let b = pulleys[(i + 1) % pulleys.len()];
            let delta = b.center - a.center;
            let d = delta.length();
            let u = delta / d;
            let ratio = (a.radius - b.radius) / d;
            let n = u * ratio + Vec3::new(0.0, u.z, -u.y) * (1.0 - ratio * ratio).sqrt();
            if pulleys
                .iter()
                .any(|p| (p.center - a.center).dot(n) + p.radius > a.radius + 1e-9 * d)
            {
                return Err(invalid());
            }
            normals.push(n);
        }
        let mut pieces = Vec::with_capacity(pulleys.len() * 2);
        let mut total_sweep = 0.0;
        for i in 0..pulleys.len() {
            let previous = normals[(i + pulleys.len() - 1) % pulleys.len()];
            let next = normals[i];
            let start = previous.z.atan2(previous.y);
            let sweep = (next.z.atan2(next.y) - start).rem_euclid(std::f64::consts::TAU);
            total_sweep += sweep;
            if sweep > 0.0 {
                pieces.push(Piece::Arc {
                    pulley: pulleys[i],
                    start,
                    sweep,
                });
            }
            pieces.push(Piece::Line {
                start: pulleys[i].center + next * pulleys[i].radius,
                end: pulleys[(i + 1) % pulleys.len()].center
                    + next * pulleys[(i + 1) % pulleys.len()].radius,
                outward: next,
            });
        }
        let length: Real = pieces.iter().map(Piece::length).sum();
        if !length.is_finite() || (total_sweep - std::f64::consts::TAU).abs() > 1e-8 {
            return Err(invalid());
        }
        Ok(Self { pieces, length })
    }

    pub fn length(&self) -> Real {
        self.length
    }

    /// Return nearest arc distance and Euclidean separation without constraining simulated nodes.
    pub fn project(&self, point: Vec3) -> Result<(Real, Real)> {
        if !point.is_finite() {
            return Err(Error::Build("non-finite belt projection".into()));
        }
        let mut base = 0.0;
        let mut best = (0.0, Real::INFINITY);
        for piece in &self.pieces {
            let length = piece.length();
            let local = match piece {
                Piece::Line { start, end, .. } => {
                    ((point - *start).dot(*end - *start) / (length * length)).clamp(0.0, 1.0)
                        * length
                }
                Piece::Arc {
                    pulley,
                    start,
                    sweep,
                } => {
                    let p = point - pulley.center;
                    let angle = (p.z.atan2(p.y) - start).rem_euclid(std::f64::consts::TAU);
                    if angle <= *sweep {
                        angle * pulley.radius
                    } else {
                        let a = pulley.center
                            + Vec3::new(0.0, start.cos(), start.sin()) * pulley.radius;
                        let end = start + sweep;
                        let b =
                            pulley.center + Vec3::new(0.0, end.cos(), end.sin()) * pulley.radius;
                        if (point - a).length_squared() <= (point - b).length_squared() {
                            0.0
                        } else {
                            length
                        }
                    }
                }
            };
            let s = (base + local).rem_euclid(self.length);
            let separation = (self.sample(s)?.position - point).length();
            if separation < best.1 {
                best = (s, separation);
            }
            base += length;
        }
        Ok(best)
    }

    pub fn sample(&self, distance: Real) -> Result<BeltPathSample> {
        if !distance.is_finite() {
            return Err(Error::Build("non-finite belt path distance".into()));
        }
        let mut s = distance.rem_euclid(self.length);
        for piece in &self.pieces {
            let length = piece.length();
            if s <= length {
                return Ok(match piece {
                    Piece::Arc { pulley, start, .. } => {
                        let angle = start + s / pulley.radius;
                        let outward = Vec3::new(0.0, angle.cos(), angle.sin());
                        BeltPathSample {
                            position: pulley.center + outward * pulley.radius,
                            tangent: Vec3::new(0.0, -outward.z, outward.y),
                            outward,
                        }
                    }
                    Piece::Line {
                        start,
                        end,
                        outward,
                    } => BeltPathSample {
                        position: *start + (*end - *start) * (s / length),
                        tangent: (*end - *start) / length,
                        outward: *outward,
                    },
                });
            }
            s -= length;
        }
        Err(Error::Build("belt path sampling exceeded perimeter".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stadium_has_exact_perimeter_and_periodic_tangent() {
        let path = BeltPath::around_pulleys(&[
            BeltPulley {
                center: Vec3::new(0.0, -0.5, 0.0),
                radius: 0.15,
            },
            BeltPulley {
                center: Vec3::new(0.0, 0.5, 0.0),
                radius: 0.15,
            },
        ])
        .unwrap();
        assert!((path.length() - (2.0 + std::f64::consts::TAU * 0.15)).abs() < 1e-12);
        for s in [-0.1, 0.0, 0.17, 1.6] {
            let a = path.sample(s).unwrap();
            let b = path.sample(s + path.length()).unwrap();
            assert!((a.position - b.position).length() < 1e-12);
            assert!((a.tangent - b.tangent).length() < 1e-12);
            assert!(a.outward.dot(a.tangent).abs() < 1e-12);
        }
    }

    #[test]
    fn three_wheel_path_is_tangent_and_outside_every_pulley() {
        let pulleys = [
            BeltPulley {
                center: Vec3::new(0.0, -0.52, -0.11),
                radius: 0.144,
            },
            BeltPulley {
                center: Vec3::new(0.0, 0.54, -0.11),
                radius: 0.144,
            },
            BeltPulley {
                center: Vec3::new(0.0, 0.405, 0.194),
                radius: 0.133,
            },
        ];
        let path = BeltPath::around_pulleys(&pulleys).unwrap();
        for i in 0..2000 {
            let s = path.length() * i as Real / 2000.0;
            let a = path.sample(s).unwrap();
            let before = path.sample(s - 1e-7).unwrap().position;
            let after = path.sample(s + 1e-7).unwrap().position;
            assert!(((after - before).normalize() - a.tangent).length() < 2e-6);
            let (projected, error) = path.project(a.position + a.outward * 0.001).unwrap();
            let ds = (projected - s + path.length() * 0.5).rem_euclid(path.length())
                - path.length() * 0.5;
            assert!(ds.abs() < 1e-10 && (error - 0.001).abs() < 1e-10);
            for p in pulleys {
                assert!((a.position - p.center).length() >= p.radius - 1e-12);
            }
        }
        assert!(BeltPath::around_pulleys(&[pulleys[2], pulleys[1], pulleys[0]]).is_err());
        assert!(BeltPath::around_pulleys(&[pulleys[0], pulleys[0]]).is_err());
        assert!(path.sample(Real::NAN).is_err());
    }
    #[test]
    fn convex_polyline_inset_preserves_distances_and_periodicity() {
        let points = [
            Vec3::new(0.0, -2.0, -1.0),
            Vec3::new(0.0, 2.0, -1.0),
            Vec3::new(0.0, 2.0, 1.0),
            Vec3::new(0.0, -2.0, 1.0),
        ];
        let outer = BeltPath::from_closed_polyline(&points, 0.0).unwrap();
        let inner = BeltPath::from_closed_polyline(&points, 0.25).unwrap();
        assert_eq!(outer.length(), 12.0);
        assert_eq!(inner.length(), 10.0);
        for i in 0..100 {
            let s = i as Real * inner.length() / 100.0 + 0.013;
            let sample = inner.sample(s).unwrap();
            let repeated = inner.sample(s - inner.length()).unwrap();
            assert!((sample.position - repeated.position).length() < 1e-12);
            assert!((outer.project(sample.position).unwrap().1 - 0.25).abs() < 1e-12);
            assert!((sample.tangent.length() - 1.0).abs() < 1e-12);
            assert!(sample.outward.dot(sample.tangent).abs() < 1e-12);
        }
        assert!(BeltPath::from_closed_polyline(&points, 1.0).is_err());
        assert!(BeltPath::from_closed_polyline(&points, 2.5).is_err());
        assert!(BeltPath::from_closed_polyline(&points, -0.1).is_err());
        assert!(BeltPath::from_closed_polyline(&points, Real::NAN).is_err());
        let mut reversed = points;
        reversed.reverse();
        assert!(BeltPath::from_closed_polyline(&reversed, 0.0).is_err());
        let mut concave = points;
        concave[1] = Vec3::new(0.0, -1.9, 0.9);
        assert!(BeltPath::from_closed_polyline(&concave, 0.0).is_err());
        let mut duplicate = points;
        duplicate[1] = duplicate[0];
        assert!(BeltPath::from_closed_polyline(&duplicate, 0.0).is_err());
        let mut nonplanar = points;
        nonplanar[1].x = 0.1;
        assert!(BeltPath::from_closed_polyline(&nonplanar, 0.0).is_err());
    }
}
