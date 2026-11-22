use super::{Bond, BondMaterial, BondPatch, Fiber, Pin, StrandModel, invalid};
use molla_core::Result;
use molla_geometry::convex_cut::ConvexCell;
use molla_math::{Quat, Real, Transform, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct BundleDesc {
    pub start: Vec3,
    pub end: Vec3,
    pub strand_radius: Real,
    pub rings: usize,
    pub segments: usize,
    pub density: Real,
    pub young_modulus: Real,
    pub shear_modulus: Real,
    pub tensile_strength: Real,
    pub shear_strength: Real,
    pub interface_ratio: Real,
    pub damage_rate: Real,
    pub damping: Real,
    pub pin_start: bool,
}

pub struct FiberBundle {
    pub model: StrandModel,
    /// Ordered cell IDs for each longitudinal strand, before any cuts.
    pub strands: Vec<Vec<u64>>,
}

impl BundleDesc {
    /// Build a hexagonally tiled cross-section with staggered interior segment boundaries.
    pub fn build(self) -> Result<FiberBundle> {
        let d = self.end - self.start;
        let length = d.length();
        if !self.start.is_finite()
            || !self.end.is_finite()
            || !length.is_finite()
            || length <= 0.0
            || self.rings > 3
            || !(3..=512).contains(&self.segments)
            || [
                self.strand_radius,
                self.density,
                self.young_modulus,
                self.shear_modulus,
                self.tensile_strength,
                self.shear_strength,
                self.interface_ratio,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
            || !self.damage_rate.is_finite()
            || self.damage_rate < 0.0
            || !self.damping.is_finite()
            || self.damping < 0.0
        {
            return Err(invalid("invalid fiber bundle description"));
        }
        let rotation = Quat::from_rotation_arc(Vec3::Z, d / length);
        let root = Transform::new(self.start, rotation);
        let mut offsets = Vec::new();
        let rings = self.rings as i32;
        for q in -rings..=rings {
            for r in (-rings).max(-q - rings)..=rings.min(-q + rings) {
                offsets.push(
                    Vec3::new(
                        1.5 * q as Real,
                        3.0_f64.sqrt() * (r as Real + 0.5 * q as Real),
                        0.0,
                    ) * self.strand_radius,
                );
            }
        }
        offsets.sort_by(|a, b| {
            a.length_squared()
                .total_cmp(&b.length_squared())
                .then(a.x.total_cmp(&b.x))
                .then(a.y.total_cmp(&b.y))
        });
        let unit = ConvexCell::prism(self.strand_radius, 1.0, 6)?.mass_properties(1.0)?;
        let area = unit.mass;
        let polar = unit.inertia.z_axis.z;
        let moment = polar * 0.5;
        let spacing = length / self.segments as Real;
        let mut fibers = Vec::new();
        let mut strands = Vec::new();
        let mut intervals = Vec::new();
        for (strand, offset) in offsets.iter().enumerate() {
            let phase = ((strand * 37) % 97) as Real / 97.0 * 0.65;
            let mut nodes = vec![0.0, spacing];
            for i in 2..self.segments - 1 {
                nodes.push((i as Real + phase) * spacing);
            }
            nodes.extend([length - spacing, length]);
            let mut ids = Vec::new();
            let mut ranges = Vec::new();
            for pair in nodes.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                let rest = Transform::new(
                    root.transform_point(*offset + Vec3::Z * ((a + b) * 0.5)),
                    rotation,
                );
                let mut fiber = Fiber::new(
                    fibers.len() as u64,
                    ConvexCell::prism(self.strand_radius, b - a, 6)?,
                    self.density,
                    rest,
                )?;
                if a == 0.0 && self.pin_start {
                    fiber.pin = Some(Pin {
                        local_point: -Vec3::Z * (0.5 * (b - a)),
                        pose: rest,
                    });
                }
                ids.push(fiber.id);
                ranges.push((a, b));
                fibers.push(fiber);
            }
            strands.push(ids);
            intervals.push(ranges);
        }
        let mut bonds = Vec::new();
        for (ids, ranges) in strands.iter().zip(&intervals) {
            for i in 1..ids.len() {
                let (a, b) = (&fibers[ids[i - 1] as usize], &fibers[ids[i] as usize]);
                let la = ranges[i - 1].1 - ranges[i - 1].0;
                let lb = ranges[i].1 - ranges[i].0;
                let dual = 0.5 * (la + lb);
                let material = BondMaterial {
                    translation: Vec3::new(
                        dual / (self.shear_modulus * area),
                        dual / (self.shear_modulus * area),
                        dual / (self.young_modulus * area),
                    ),
                    rotation: Vec3::new(
                        dual / (self.young_modulus * moment),
                        dual / (self.young_modulus * moment),
                        dual / (self.shear_modulus * polar),
                    ),
                    force_limit: Vec3::new(
                        self.shear_strength * area,
                        self.shear_strength * area,
                        self.tensile_strength * area,
                    ),
                    torque_limit: Vec3::new(
                        self.tensile_strength * moment,
                        self.tensile_strength * moment,
                        self.shear_strength * polar,
                    ) / self.strand_radius,
                    damage_rate: self.damage_rate,
                    damping: self.damping,
                };
                let patch = BondPatch::new(
                    (0..6)
                        .map(|i| {
                            let angle = i as Real * std::f64::consts::TAU / 6.0;
                            let xy = Vec3::new(angle.cos(), angle.sin(), 0.0) * self.strand_radius;
                            [xy + Vec3::Z * (0.5 * la), xy - Vec3::Z * (0.5 * lb)]
                        })
                        .collect(),
                )?;
                bonds.push(
                    Bond::new(
                        bonds.len() as u64,
                        [a, b],
                        [Vec3::Z * (0.5 * la), -Vec3::Z * (0.5 * lb)],
                        material,
                    )?
                    .with_patch(patch)?,
                );
            }
        }
        for i in 0..strands.len() {
            for j in i + 1..strands.len() {
                if (offsets[i].distance(offsets[j]) - 3.0_f64.sqrt() * self.strand_radius).abs()
                    > self.strand_radius * 1e-8
                {
                    continue;
                }
                let mut a = 0;
                let mut b = 0;
                while a < strands[i].len() && b < strands[j].len() {
                    let ra = intervals[i][a];
                    let rb = intervals[j][b];
                    let lo = ra.0.max(rb.0);
                    let hi = ra.1.min(rb.1);
                    if hi - lo > length * 1e-10 {
                        let ia = strands[i][a] as usize;
                        let ib = strands[j][b] as usize;
                        let point = root.transform_point(
                            (offsets[i] + offsets[j]) * 0.5 + Vec3::Z * ((lo + hi) * 0.5),
                        );
                        let patch_area = (hi - lo) * self.strand_radius;
                        let compliance = 3.0_f64.sqrt() * self.strand_radius
                            / (self.shear_modulus * patch_area * self.interface_ratio);
                        let strength = self.shear_strength * patch_area * self.interface_ratio;
                        let normal = (offsets[j] - offsets[i]).normalize();
                        let tangent = root.transform_vector(Vec3::new(-normal.y, normal.x, 0.0))
                            * self.strand_radius
                            * 0.5;
                        let axial = root.transform_vector(Vec3::Z) * (hi - lo) * 0.5;
                        let patch = BondPatch::new(
                            [
                                point - tangent - axial,
                                point + tangent - axial,
                                point + tangent + axial,
                                point - tangent + axial,
                            ]
                            .map(|p| {
                                [
                                    fibers[ia].rest.inverse_transform_point(p),
                                    fibers[ib].rest.inverse_transform_point(p),
                                ]
                            })
                            .to_vec(),
                        )?;
                        let material = BondMaterial {
                            translation: Vec3::splat(compliance),
                            rotation: Vec3::splat(compliance / self.strand_radius.powi(2)),
                            force_limit: Vec3::splat(strength),
                            torque_limit: Vec3::splat(strength * self.strand_radius),
                            damage_rate: self.damage_rate,
                            damping: self.damping,
                        };
                        bonds.push(
                            Bond::new(
                                bonds.len() as u64,
                                [&fibers[ia], &fibers[ib]],
                                [
                                    fibers[ia].rest.inverse_transform_point(point),
                                    fibers[ib].rest.inverse_transform_point(point),
                                ],
                                material,
                            )?
                            .with_patch(patch)?,
                        );
                    }
                    if ra.1 <= rb.1 {
                        a += 1;
                    } else {
                        b += 1;
                    }
                }
            }
        }
        Ok(FiberBundle {
            model: StrandModel::new(fibers, bonds)?,
            strands,
        })
    }
}
