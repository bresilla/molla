//! Pressure-dependent brush tyres in a local planar contact frame.

use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_vehicle::real_tire::{
    RealTireParams, RealWheelState, SurfaceMaterial, TireBrushState, TireContactCell,
    evaluate_brush_tire, evaluate_planar_brush_tire_persistent,
};
use molla_vehicle::{FlatTerrain, TireOutput, WheelContact};

#[derive(Clone, Copy, Debug)]
pub struct PressureTireConfig {
    /// Reference parameters; inflation pressure is gauge pressure in Pa.
    pub tire: RealTireParams,
    pub min_pressure_pa: Real,
    pub max_pressure_pa: Real,
    pub pressure_rate_pa_s: Real,
    pub supported_mass: Real,
    pub damping_ratio: Real,
    /// Carcass hysteresis moment divided by normal load times deflection.
    pub hysteresis_fraction: Real,
}

impl PressureTireConfig {
    pub fn validate(&self) -> Result<()> {
        let t = &self.tire;
        let positive = [
            t.radius,
            t.width,
            t.carcass_normal_stiffness,
            t.relaxation_speed,
            self.pressure_rate_pa_s,
            self.supported_mass,
        ];
        let nonnegative = [
            t.pressure_pa,
            t.tread_shear_stiffness,
            self.min_pressure_pa,
            self.max_pressure_pa,
            self.damping_ratio,
            self.hysteresis_fraction,
        ];
        if positive.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || nonnegative.iter().any(|v| !v.is_finite() || *v < 0.0)
            || !(8..=512).contains(&t.segments)
            || t.radius < 1e-4
            || t.width < 1e-4
            || t.relaxation_speed < 1e-4
            || t.carcass_normal_stiffness < 1.0
            || !t.structural_lateral_bias_per_normal.is_finite()
            || t.structural_lateral_bias_per_normal.abs() > 0.08
            || self.min_pressure_pa > self.max_pressure_pa
            || !(self.min_pressure_pa..=self.max_pressure_pa).contains(&t.pressure_pa)
        {
            return Err(Error::Build("invalid pressure tyre parameters".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PressureTireState {
    pub config: PressureTireConfig,
    pub pressure_pa: Real,
    pub target_pressure_pa: Real,
}

impl PressureTireState {
    pub(crate) fn advance(&mut self, dt: Real) {
        let step = self.config.pressure_rate_pa_s * dt;
        self.pressure_pa += (self.target_pressure_pa - self.pressure_pa).clamp(-step, step);
    }
}

#[derive(Clone, Debug)]
pub struct PressureTireSample {
    pub pressure_pa: Real,
    pub target_pressure_pa: Real,
    pub deflection: Real,
    pub loaded_radius: Real,
    pub patch_length: Real,
    pub patch_width: Real,
    /// Sum of active tread-cell areas, before planar projection.
    pub patch_area: Real,
    pub normal_stiffness: Real,
    pub grip_force: Real,
    /// Applied world-space resisting moment, separate from contact-force torque.
    pub rolling_moment: Vec3,
    pub cells: Vec<TireContactCell>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PressureShear {
    brush: TireBrushState,
    frame: Option<(Vec3, Vec3)>,
}

#[cfg(test)]
pub(crate) fn evaluate(
    state: &PressureTireState,
    wheel: &WheelContact,
    normal: Vec3,
    distance: Real,
    friction: Real,
) -> Result<(TireOutput, PressureTireSample)> {
    evaluate_with_shear(state, wheel, normal, distance, friction, None)
}

pub(crate) fn evaluate_with_shear(
    state: &PressureTireState,
    wheel: &WheelContact,
    normal: Vec3,
    distance: Real,
    friction: Real,
    shear: Option<(&mut PressureShear, Real)>,
) -> Result<(TireOutput, PressureTireSample)> {
    let mut params = state.config.tire;
    params.pressure_pa = state.pressure_pa;
    let heading = Vec3::from_array(wheel.heading);
    let forward = (heading - normal * heading.dot(normal)).normalize_or_zero();
    if forward.length_squared() < 0.5 {
        return Err(Error::Solver("degenerate pressure tyre heading".into()));
    }
    let side = forward.cross(normal);
    let velocity = Vec3::from_array(wheel.hub_vel);
    let loaded_radius = distance.clamp(0.0, params.radius);
    let deflection = params.radius - loaded_radius;
    let local = RealWheelState {
        hub_pos: [0.0, distance, 0.0],
        hub_vel: [
            velocity.dot(forward),
            velocity.dot(normal),
            velocity.dot(side),
        ],
        heading: [1.0, 0.0, 0.0],
        omega: wheel.spin * loaded_radius / params.radius,
    };
    let mut material = SurfaceMaterial::from_terrain_friction(friction);
    let rolling_coefficient = material.rolling_resistance;
    material.rolling_resistance = 0.0;
    let mut contact = if let Some((shear, dt)) = shear {
        if let Some((old_forward, old_side)) = shear.frame {
            shear
                .brush
                .project_frame([
                    [forward.dot(old_forward), forward.dot(old_side)],
                    [side.dot(old_forward), side.dot(old_side)],
                ])
                .map_err(|e| Error::Solver(e.into()))?;
        }
        shear.frame = Some((forward, side));
        evaluate_planar_brush_tire_persistent(
            &params,
            &local,
            &FlatTerrain::new(0.0, friction),
            &material,
            &mut shear.brush,
            dt,
        )
        .map_err(|e| Error::Solver(e.into()))?
    } else {
        evaluate_brush_tire(&params, &local, &FlatTerrain::new(0.0, friction), &material)
    };
    if !contact.is_finite() {
        return Err(Error::Solver("non-finite pressure tyre contact".into()));
    }
    let area: Real = contact.cells.iter().map(|c| c.area).sum();
    let stiffness =
        area * (params.carcass_normal_stiffness + 3.0 * params.pressure_pa / params.radius);
    let damping =
        2.0 * state.config.damping_ratio * (stiffness * state.config.supported_mass).sqrt();
    let normal_force = if contact.in_contact {
        (contact.normal_load - damping * local.hub_vel[1]).max(0.0)
    } else {
        0.0
    };
    let load_scale = if contact.normal_load > 0.0 {
        normal_force / contact.normal_load
    } else {
        0.0
    };
    for cell in &mut contact.cells {
        cell.normal = normal.to_array();
        cell.normal_load *= load_scale;
        cell.tangential_force *= load_scale;
        cell.lateral_force *= load_scale;
    }
    let force = normal * normal_force
        + load_scale * (forward * contact.longitudinal_force + side * contact.lateral_force);
    let hub = Vec3::from_array(wheel.hub_pos);
    let point = hub - normal * loaded_radius;
    let magnitude = normal_force
        * (rolling_coefficient * loaded_radius + state.config.hysteresis_fraction * deflection);
    let rolling_moment =
        side * (magnitude * (wheel.spin * loaded_radius / params.relaxation_speed).tanh());
    if !force.is_finite() || !rolling_moment.is_finite() || !stiffness.is_finite() {
        return Err(Error::Solver("non-finite pressure tyre wrench".into()));
    }
    let output = TireOutput {
        in_contact: normal_force > 0.0,
        force: force.to_array(),
        contact_point: point.to_array(),
        aligning_moment: contact.aligning_moment_z * load_scale,
        fz: normal_force,
        slip_ratio: contact.slip_ratio,
        slip_angle: -contact.slip_angle_rad,
    };
    let sample = PressureTireSample {
        pressure_pa: state.pressure_pa,
        target_pressure_pa: state.target_pressure_pa,
        deflection,
        loaded_radius,
        patch_length: contact.patch_length,
        patch_width: if area > 0.0 { params.width } else { 0.0 },
        patch_area: area,
        normal_stiffness: stiffness,
        grip_force: contact.available_friction_budget * load_scale,
        rolling_moment,
        cells: contact.cells,
    };
    if !sample.grip_force.is_finite() || !output.aligning_moment.is_finite() {
        return Err(Error::Solver("non-finite pressure tyre readout".into()));
    }
    Ok((output, sample))
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_math::Quat;
    use molla_vehicle::real_tire::solve_loaded_tire_hub_height;

    fn state(pressure_pa: Real) -> PressureTireState {
        PressureTireState {
            config: PressureTireConfig {
                tire: RealTireParams {
                    radius: 0.5,
                    width: 0.3,
                    pressure_pa,
                    segments: 128,
                    ..Default::default()
                },
                min_pressure_pa: 50_000.0,
                max_pressure_pa: 400_000.0,
                pressure_rate_pa_s: 10_000.0,
                supported_mass: 800.0,
                damping_ratio: 0.7,
                hysteresis_fraction: 0.1,
            },
            pressure_pa,
            target_pressure_pa: pressure_pa,
        }
    }

    fn loaded(pressure: Real, slip: Real) -> (TireOutput, PressureTireSample) {
        let state = state(pressure);
        let height = solve_loaded_tire_hub_height(
            &state.config.tire,
            &FlatTerrain::new(0.0, 0.8),
            0.0,
            0.0,
            8000.0,
        );
        let contact = WheelContact {
            hub_pos: [0.0, height, 0.0],
            hub_vel: [2.0, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            spin: 2.0 * (1.0 + slip) / height,
            radius: 0.5,
            normal_stiffness: 0.0,
            normal_damping: 0.0,
        };
        evaluate(&state, &contact, Vec3::Y, height, 0.8).unwrap()
    }

    #[test]
    fn persistent_patch_holds_shear_rotates_frames_and_clears_airborne() {
        let state = state(180_000.0);
        let mut history = PressureShear::default();
        let mut wheel = WheelContact {
            hub_pos: [0.0, 0.46, 0.0],
            hub_vel: [0.1, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            spin: 0.0,
            radius: 0.5,
            normal_stiffness: 0.0,
            normal_damping: 0.0,
        };
        evaluate_with_shear(
            &state,
            &wheel,
            Vec3::Y,
            0.46,
            0.8,
            Some((&mut history, 0.01)),
        )
        .unwrap();
        wheel.hub_vel = [0.0; 3];
        let held = evaluate_with_shear(
            &state,
            &wheel,
            Vec3::Y,
            0.46,
            0.8,
            Some((&mut history, 0.01)),
        )
        .unwrap()
        .0;
        assert!(held.force[0] < -10.0);
        for _ in 0..7200 {
            let out = evaluate_with_shear(
                &state,
                &wheel,
                Vec3::Y,
                0.46,
                0.8,
                Some((&mut history, 1.0 / 120.0)),
            )
            .unwrap()
            .0;
            assert_eq!(out.force, held.force);
        }
        wheel.heading = [0.0, 0.0, 1.0];
        let rotated = evaluate_with_shear(
            &state,
            &wheel,
            Vec3::Y,
            0.46,
            0.8,
            Some((&mut history, 0.01)),
        )
        .unwrap()
        .0;
        assert!((Vec3::from_array(rotated.force) - Vec3::from_array(held.force)).length() < 1e-8);
        let airborne = evaluate_with_shear(
            &state,
            &wheel,
            Vec3::Y,
            0.6,
            0.8,
            Some((&mut history, 0.01)),
        )
        .unwrap()
        .0;
        assert!(!airborne.in_contact);
        let landed = evaluate_with_shear(
            &state,
            &wheel,
            Vec3::Y,
            0.46,
            0.8,
            Some((&mut history, 0.01)),
        )
        .unwrap()
        .0;
        assert_eq!(landed.force[0], 0.0);
        assert_eq!(landed.force[2], 0.0);
    }

    #[test]
    fn matched_load_pressure_changes_footprint_grip_and_hysteresis() {
        let (low, lp) = loaded(80_000.0, 0.15);
        let (high, hp) = loaded(320_000.0, 0.15);
        for (output, sample) in [(&low, &lp), (&high, &hp)] {
            assert!((output.fz - 8000.0).abs() < 1e-7);
            assert!(
                (sample.cells.iter().map(|c| c.normal_load).sum::<Real>() - output.fz).abs() < 1e-8
            );
            assert!((output.slip_ratio - 0.15).abs() < 1e-10);
            assert!(sample.grip_force <= 1.12 * 0.8 * output.fz + 1e-8);
        }
        assert!(lp.deflection > hp.deflection);
        assert!(lp.loaded_radius < hp.loaded_radius);
        assert!(lp.patch_area > hp.patch_area);
        assert!(lp.patch_length > hp.patch_length);
        assert!((lp.grip_force - hp.grip_force).abs() > 1.0);
        assert!(lp.rolling_moment.length() > hp.rolling_moment.length());
    }

    #[test]
    fn loaded_radius_has_zero_slip_and_separate_resisting_moment() {
        for pressure in [80_000.0, 180_000.0, 320_000.0] {
            let (output, sample) = loaded(pressure, 0.0);
            assert!(output.slip_ratio.abs() < 1e-12);
            assert!(output.force[0].abs() < 1e-8);
            assert!(sample.rolling_moment.dot(-Vec3::Z) < 0.0);
        }
    }

    #[test]
    fn planar_contact_rotates_with_the_slope() {
        let state = state(180_000.0);
        let wheel = WheelContact {
            hub_pos: [0.0, 0.46, 0.0],
            hub_vel: [2.0, -0.1, 0.2],
            heading: [1.0, 0.0, 0.0],
            spin: 5.0,
            radius: 0.5,
            normal_stiffness: 0.0,
            normal_damping: 0.0,
        };
        let (flat, fp) = evaluate(&state, &wheel, Vec3::Y, 0.46, 0.8).unwrap();
        let rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.15, 0.3, -0.2);
        let tilted = WheelContact {
            hub_pos: (rotation * Vec3::from_array(wheel.hub_pos)).to_array(),
            hub_vel: (rotation * Vec3::from_array(wheel.hub_vel)).to_array(),
            heading: (rotation * Vec3::X).to_array(),
            ..wheel
        };
        let (slope, sp) = evaluate(&state, &tilted, rotation * Vec3::Y, 0.46, 0.8).unwrap();
        assert!(
            (Vec3::from_array(slope.force) - rotation * Vec3::from_array(flat.force)).length()
                < 1e-8
        );
        assert!(
            (Vec3::from_array(slope.contact_point)
                - rotation * Vec3::from_array(flat.contact_point))
            .length()
                < 1e-10
        );
        assert!((sp.rolling_moment - rotation * fp.rolling_moment).length() < 1e-8);
        assert!((slope.slip_angle - flat.slip_angle).abs() < 1e-10);
        for cell in &sp.cells {
            assert!((Vec3::from_array(cell.normal) - rotation * Vec3::Y).length() < 1e-10);
        }
    }
}
