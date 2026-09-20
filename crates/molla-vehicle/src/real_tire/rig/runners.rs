//! Tire-rig step-response + sweep + brush-evaluation runners.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use super::super::{TireBrushState, TreadBristle, TreadBristleParams};
use crate::terrain::Terrain;

use super::super::internal::{
    add, cross, dot, finite_nonnegative, finite_or, finite_positive_or, finite_vec3, normalize,
    scale, sub,
};
use super::super::surface_material::sanitized_surface_material;
use super::super::{
    RealTireContact, RealTireParams, RealWheelOrientation, RealWheelState, SurfaceFilmState,
    SurfaceMaterial, TireContactCell, TireRigStepResponseConfig, TireRigStepResponsePoint,
    TireRigSweepPoint, TireWearState, brush_bristle_profile, brush_load_sensitivity,
    brush_pressure_patch_gain, brush_pressure_stiffness_factor, project_to_ground,
    sanitize_brush_params, smooth_combined_brush_force, tire_rig_point, tire_rig_step_response,
};

/// Deterministic drive-torque step fixture.
pub fn tire_rig_step_torque_response(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    load_n: f64,
    speed_m_per_s: f64,
    drive_torque_after_nm: f64,
    dt_s: f64,
    steps: usize,
) -> Vec<TireRigStepResponsePoint> {
    tire_rig_step_response(
        params,
        terrain,
        material,
        TireRigStepResponseConfig {
            load_n,
            speed_m_per_s,
            drive_torque_after_nm,
            dt_s,
            steps,
            ..Default::default()
        },
    )
}

/// Deterministic brake-torque step fixture.
pub fn tire_rig_step_brake_response(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    load_n: f64,
    speed_m_per_s: f64,
    brake_torque_after_nm: f64,
    dt_s: f64,
    steps: usize,
) -> Vec<TireRigStepResponsePoint> {
    tire_rig_step_response(
        params,
        terrain,
        material,
        TireRigStepResponseConfig {
            load_n,
            speed_m_per_s,
            brake_torque_after_nm,
            dt_s,
            steps,
            ..Default::default()
        },
    )
}

/// Deterministic steer-angle step fixture.
pub fn tire_rig_step_steer_response(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    load_n: f64,
    speed_m_per_s: f64,
    steer_angle_after_rad: f64,
    dt_s: f64,
    steps: usize,
) -> Vec<TireRigStepResponsePoint> {
    tire_rig_step_response(
        params,
        terrain,
        material,
        TireRigStepResponseConfig {
            load_n,
            speed_m_per_s,
            steer_angle_after_rad,
            dt_s,
            steps,
            ..Default::default()
        },
    )
}

/// Deterministic combined longitudinal/lateral slip grid for tire-rig
/// validation. Points are emitted with slip ratio as the outer loop and slip
/// angle as the inner loop.
pub fn tire_rig_combined_slip_sweep(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    load_n: f64,
    speed_m_per_s: f64,
    slip_ratios: &[f64],
    slip_angles_rad: &[f64],
) -> Vec<TireRigSweepPoint> {
    let mut points = Vec::with_capacity(slip_ratios.len() * slip_angles_rad.len());
    for &slip in slip_ratios {
        for &angle in slip_angles_rad {
            points.push(tire_rig_point(
                params,
                terrain,
                material,
                load_n,
                speed_m_per_s,
                slip,
                angle,
                0.0,
            ));
        }
    }
    points
}

/// Deterministic camber sweep for tire-rig validation.
pub fn tire_rig_camber_sweep(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    load_n: f64,
    speed_m_per_s: f64,
    camber_angles_rad: &[f64],
) -> Vec<TireRigSweepPoint> {
    camber_angles_rad
        .iter()
        .map(|&camber| {
            tire_rig_point(
                params,
                terrain,
                material,
                load_n,
                speed_m_per_s,
                0.0,
                0.0,
                camber,
            )
        })
        .collect()
}

pub(crate) fn surface_material_with_terrain_film(
    material: &SurfaceMaterial,
    terrain: &dyn Terrain,
    x: f64,
    z: f64,
    speed_mps: f64,
) -> SurfaceMaterial {
    let base = sanitized_surface_material(material);
    let water_depth_m = finite_nonnegative(terrain.water_film_depth(x, z));
    if water_depth_m <= 1.0e-7 {
        return base;
    }
    SurfaceFilmState::wet(water_depth_m).apply_to(base, speed_mps, &TireWearState::default())
}

/// Solve the hub height that gives `target_load` on flat or heightfield-like
/// terrain at the current `(x,z)` location.
pub fn solve_loaded_tire_hub_height(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    x: f64,
    z: f64,
    target_load: f64,
) -> f64 {
    let sp = sanitize_brush_params(params);
    let gh = terrain.height(x, z);
    let target_load = finite_nonnegative(target_load);
    let mut lo = gh + sp.radius * 0.18;
    let mut hi = gh + sp.radius * 1.08;
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        let state = RealWheelState {
            hub_pos: [x, mid, z],
            hub_vel: [0.0; 3],
            heading: [1.0, 0.0, 0.0],
            omega: 0.0,
        };
        let out = evaluate_brush_tire(
            params,
            &state,
            terrain,
            &SurfaceMaterial::from_terrain_friction(terrain.friction()),
        );
        if out.normal_load > target_load {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Evaluate the pressure-dependent contact patch and longitudinal brush force.
pub fn evaluate_brush_tire(
    params: &RealTireParams,
    state: &RealWheelState,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
) -> RealTireContact {
    evaluate_brush_tire_oriented(
        params,
        state,
        terrain,
        material,
        &RealWheelOrientation::default(),
    )
}

/// Evaluate brush tire contact with an explicit wheel orientation.
///
/// Use this when camber matters. `evaluate_brush_tire` remains the
/// zero-camber compatibility entrypoint.
pub fn evaluate_brush_tire_oriented(
    params: &RealTireParams,
    state: &RealWheelState,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    orientation: &RealWheelOrientation,
) -> RealTireContact {
    evaluate_brush_impl(params, state, terrain, material, orientation, None)
        .unwrap_or_else(|_| RealTireContact::empty())
}

/// Evaluate a planar patch with persistent cell shear; errors retain the previous state.
/// Terrain friction and film properties may vary across the planar contact patch.
/// The caller projects history when the tangent frame changes.
pub fn evaluate_planar_brush_tire_persistent(
    params: &RealTireParams,
    state: &RealWheelState,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    history: &mut TireBrushState,
    dt: f64,
) -> Result<RealTireContact, &'static str> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err("invalid brush timestep");
    }
    let mut next = history.clone();
    let sanitized = sanitize_brush_params(params);
    let geometry = (sanitized.radius, sanitized.width, sanitized.segments, sanitized.width_segments);
    if next.geometry != Some(geometry) {
        next.reset();
        next.geometry = Some(geometry);
        next.profile = brush_profile(sanitized.radius, sanitized.width, sanitized.segments, sanitized.width_segments);
        next.cells
            .resize(sanitized.segments * sanitized.width_segments, TreadBristle::default());
    }
    let output = evaluate_brush_impl(
        params,
        state,
        terrain,
        material,
        &RealWheelOrientation::default(),
        Some((&mut next, dt)),
    )?;
    if !output.is_finite() {
        return Err("non-finite persistent brush output");
    }
    *history = next;
    Ok(output)
}

fn brush_profile(radius: f64, width: f64, segments: usize, width_segments: usize) -> std::sync::Arc<[[f64; 3]]> {
    (0..segments).flat_map(|i| {
        let theta = i as f64 * (std::f64::consts::TAU / segments as f64);
        let (x, y) = (radius * theta.sin(), -radius * theta.cos());
        (0..width_segments).map(move |strip| {
            [x, y, ((strip as f64 + 0.5) / width_segments as f64 - 0.5) * width]
        })
    }).collect()
}

fn evaluate_brush_impl(
    params: &RealTireParams,
    state: &RealWheelState,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    orientation: &RealWheelOrientation,
    mut persistent: Option<(&mut TireBrushState, f64)>,
) -> Result<RealTireContact, &'static str> {
    if !finite_vec3(state.hub_pos)
        || !finite_vec3(state.hub_vel)
        || !finite_vec3(state.heading)
        || !finite_vec3(orientation.wheel_up)
    {
        return Err("invalid brush kinematics");
    }
    let params = sanitize_brush_params(params);
    let material = sanitized_surface_material(material);
    let segments = params.segments;
    let width_segments = params.width_segments;
    let profile = persistent.as_ref().map_or_else(
        || brush_profile(params.radius, params.width, segments, width_segments),
        |(history, _)| history.profile.clone(),
    );
    let normal = terrain.normal(state.hub_pos[0], state.hub_pos[2]);
    let heading = normalize(project_to_ground(state.heading, normal));
    let side = normalize(cross(heading, normal));
    let wheel_up = normalize(orientation.wheel_up);
    let camber_angle_rad = dot(wheel_up, side).asin().clamp(-0.7, 0.7);
    let v_long = dot(state.hub_vel, heading);
    let v_lat = dot(state.hub_vel, side);
    let omega = finite_or(state.omega, 0.0);
    let wheel_speed = omega * params.radius;
    let slip_speed = wheel_speed - v_long;
    let denom = v_long.abs().max(params.relaxation_speed);
    let slip_ratio = slip_speed / denom;
    let lateral_slip_speed = -v_lat;
    let slip_angle_rad = v_lat.atan2(v_long.abs().max(params.relaxation_speed));
    let dtheta = std::f64::consts::TAU / segments as f64;
    let arc_len = params.radius * dtheta;
    let strip_width = params.width / width_segments as f64;
    let cell_area = strip_width * arc_len;
    let normal_per_area_per_m =
        params.carcass_normal_stiffness + 3.0 * params.pressure_pa / params.radius.max(1.0e-6);
    let terrain_mu_ref =
        finite_positive_or(terrain.friction(), material.mu_static.max(1.0e-3), 1.0e-6);

    // The linear leading-edge bristle profile needs the patch extent and
    // travel direction before forces are evaluated, so run a cheap
    // geometry-only pre-pass over the penetrating segments first.
    let mut patch_min_x = f64::INFINITY;
    let mut patch_max_x = f64::NEG_INFINITY;
    for &[local_x, local_y, local_z] in profile.iter() {
        let world_x = state.hub_pos[0] + heading[0] * local_x + side[0] * local_z;
        let world_z = state.hub_pos[2] + heading[2] * local_x + side[2] * local_z;
        if terrain.height(world_x, world_z) - (state.hub_pos[1] + local_y) > 0.0 {
            patch_min_x = patch_min_x.min(local_x);
            patch_max_x = patch_max_x.max(local_x);
        }
    }
    let patch_extent = (patch_max_x - patch_min_x).max(0.0);
    let leading_x = if v_long + wheel_speed >= 0.0 {
        patch_max_x
    } else {
        patch_min_x
    };
    let patch_gain = brush_pressure_patch_gain(
        params.carcass_normal_stiffness,
        params.pressure_pa,
        params.radius,
    );

    let mut cells = Vec::new();
    let mut normal_load = 0.0;
    let mut fx = 0.0;
    let mut fy = 0.0;
    let mut mz = 0.0;
    let mut normal_moment_x = 0.0;
    let mut lateral_moment_x = 0.0;
    let mut sliding_count = 0usize;
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut available_friction_budget = 0.0;
    let mut consumed_friction_budget = 0.0;
    let mut force_vec = [0.0; 3];
    let mut rolling_resistance_force = 0.0;

    for (cell_index, &[local_x, local_y, local_z]) in profile.iter().enumerate() {
        let world_x = state.hub_pos[0] + heading[0] * local_x + side[0] * local_z;
        let world_z = state.hub_pos[2] + heading[2] * local_x + side[2] * local_z;
        let point_y = state.hub_pos[1] + local_y;
        let penetration = terrain.height(world_x, world_z) - point_y;
        if penetration <= 0.0 {
            if let Some((history, _)) = &mut persistent {
                history.cells[cell_index].reset();
            }
            continue;
        }
        let cell_normal = normalize(terrain.normal(world_x, world_z));
        let cell_heading = normalize(project_to_ground(state.heading, cell_normal));
        let cell_side = normalize(cross(cell_heading, cell_normal));
        let ground_v = terrain.surface_velocity(world_x, world_z);
        let rel_ground_v = sub(state.hub_vel, ground_v);
        let v_long_cell = dot(rel_ground_v, cell_heading);
        let v_lat_cell = dot(rel_ground_v, cell_side);
        let denom_cell = v_long_cell.abs().max(params.relaxation_speed);
        let slip_speed_cell = wheel_speed - v_long_cell;
        let slip_ratio_cell = (slip_speed_cell / denom_cell).clamp(-8.0, 8.0);
        let lateral_slip_speed_cell = -v_lat_cell;
        let cell_material = surface_material_with_terrain_film(
            &material,
            terrain,
            world_x,
            world_z,
            (slip_speed_cell * slip_speed_cell + lateral_slip_speed_cell * lateral_slip_speed_cell)
                .sqrt(),
        );
        let terrain_mu = finite_nonnegative(terrain.friction_at(world_x, world_z));
        let friction_scale = (terrain_mu / terrain_mu_ref).clamp(0.0, 3.0);
        let mu_static_cell = cell_material.mu_static * friction_scale;
        let mu_dynamic_cell = cell_material.mu_dynamic * friction_scale;

        let camber_pressure_bias = (1.0
            + 0.35 * camber_angle_rad * (local_x / params.radius).clamp(-1.0, 1.0))
        .clamp(0.35, 1.65);
        let fn_i = normal_per_area_per_m * penetration * cell_area * camber_pressure_bias;
        let normal_pressure = fn_i / cell_area.max(1.0e-9);
        let (load_mu_factor, load_stiffness_factor) = brush_load_sensitivity(normal_pressure);
        let shear_k_eff = params.tread_shear_stiffness
            * load_stiffness_factor
            * brush_pressure_stiffness_factor(params.pressure_pa);
        let profile = patch_gain * brush_bristle_profile(local_x, leading_x, patch_extent, arc_len);
        let bristle_disp_long = slip_ratio_cell * params.radius * profile;
        let bristle_disp_lat = lateral_slip_speed_cell / denom_cell * params.radius * profile;
        let camber_thrust = -0.22 * camber_angle_rad * shear_k_eff * cell_area;
        let structural_lateral_bias = params.structural_lateral_bias_per_normal * fn_i;
        let trial_long = shear_k_eff * cell_area * bristle_disp_long
            + cell_material.bristle_damping * cell_area * slip_speed_cell;
        let trial_lat = shear_k_eff * cell_area * bristle_disp_lat
            + cell_material.bristle_damping * cell_area * lateral_slip_speed_cell
            + camber_thrust
            + structural_lateral_bias;
        let static_cap =
            mu_static_cell * load_mu_factor * fn_i + cell_material.cohesion * cell_area;
        let dynamic_cap =
            mu_dynamic_cell * load_mu_factor * fn_i + cell_material.cohesion * cell_area;
        let (ft_long, ft_lat, sliding, available_i, consumed_i) =
            if let Some((history, dt)) = &mut persistent {
                let config = TreadBristleParams {
                    stiffness: shear_k_eff * cell_area,
                    friction_limit: static_cap,
                    renewal_rate: wheel_speed.abs() / (params.radius * profile).max(arc_len * 0.5),
                };
                let velocity = [-slip_speed_cell, -lateral_slip_speed_cell];
                let previous = history.cells[cell_index];
                let mut result = history.cells[cell_index].advance(velocity, *dt, config)?;
                if result.sliding {
                    history.cells[cell_index] = previous;
                    result = history.cells[cell_index].advance(
                        velocity,
                        *dt,
                        TreadBristleParams {
                            friction_limit: dynamic_cap,
                            ..config
                        },
                    )?;
                }
                let mut force = [
                    result.force[0] + cell_material.bristle_damping * cell_area * slip_speed_cell,
                    result.force[1]
                        + cell_material.bristle_damping * cell_area * lateral_slip_speed_cell
                        + camber_thrust
                        + structural_lateral_bias,
                ];
                let magnitude = force[0].hypot(force[1]);
                let sliding = result.sliding || magnitude > static_cap;
                let limit = if sliding { dynamic_cap } else { static_cap };
                if magnitude > limit {
                    force = force.map(|f| f * limit / magnitude);
                }
                (
                    force[0],
                    force[1],
                    sliding,
                    static_cap,
                    force[0].hypot(force[1]),
                )
            } else {
                smooth_combined_brush_force(trial_long, trial_lat, static_cap, dynamic_cap)
            };
        let rolling_direction = if v_long_cell == 0.0 { 0.0 } else { v_long_cell.signum() };
        let rolling_i = cell_material.rolling_resistance * fn_i * rolling_direction;
        rolling_resistance_force += rolling_i;

        normal_load += fn_i;
        fx += ft_long;
        fy += ft_lat;
        force_vec = add(
            force_vec,
            add(
                add(
                    scale(cell_heading, ft_long - rolling_i),
                    scale(cell_side, ft_lat),
                ),
                scale(cell_normal, fn_i),
            ),
        );
        available_friction_budget += available_i;
        consumed_friction_budget += consumed_i;
        mz += local_z * (ft_long - rolling_i) - local_x * ft_lat;
        normal_moment_x += local_x * fn_i;
        lateral_moment_x += local_x * ft_lat.abs();
        if sliding {
            sliding_count += 1;
        }
        min_x = min_x.min(local_x);
        max_x = max_x.max(local_x);
        cells.push(TireContactCell {
            node_index: None,
            local_x,
            local_z,
            penetration,
            area: cell_area,
            normal_load: fn_i,
            normal: cell_normal,
            tangential_force: ft_long,
            lateral_force: ft_lat,
            sliding,
        });
    }

    if cells.is_empty() {
        return Ok(RealTireContact::empty());
    }

    let rolling = rolling_resistance_force;
    let longitudinal_force = fx - rolling;
    let lateral_force = fy;
    let sliding_fraction = sliding_count as f64 / cells.len().max(1) as f64;
    let adhesion_fraction = (1.0 - sliding_fraction).clamp(0.0, 1.0);
    let effective_longitudinal_stiffness = {
        let disp = (slip_ratio * params.radius).abs();
        if disp > 1.0e-6 { fx.abs() / disp } else { 0.0 }
    };
    let effective_cornering_stiffness = if slip_angle_rad.abs() > 1.0e-5 {
        fy.abs() / slip_angle_rad.abs()
    } else {
        0.0
    };
    let contact_center_x = if normal_load > 1.0e-9 {
        normal_moment_x / normal_load
    } else {
        0.0
    };
    let force_resultant_x = if fy.abs() > 1.0e-9 {
        lateral_moment_x / fy.abs()
    } else {
        contact_center_x
    };
    let pneumatic_trail_m = contact_center_x - force_resultant_x;
    // Stage (j) — heat generation: sum |Ft · v_slip| only over the
    // sliding cells. Cells in static friction don't dissipate
    // (they're just storing elastic energy in the bristles).
    let dissipated_power: f64 = cells
        .iter()
        .filter(|c| c.sliding)
        .map(|c| {
            let ft = (c.tangential_force * c.tangential_force + c.lateral_force * c.lateral_force)
                .sqrt();
            let vs = (slip_speed * slip_speed + lateral_slip_speed * lateral_slip_speed).sqrt();
            ft * vs
        })
        .sum();
    let force = force_vec;
    Ok(RealTireContact {
        in_contact: true,
        force,
        normal_load,
        longitudinal_force,
        lateral_force,
        aligning_moment_z: mz,
        camber_angle_rad,
        contact_center_x,
        force_resultant_x,
        pneumatic_trail_m,
        rolling_resistance_force: rolling,
        slip_ratio,
        slip_speed,
        lateral_slip_speed,
        slip_angle_rad,
        patch_length: (max_x - min_x).max(0.0),
        contact_count: cells.len(),
        sliding_count,
        adhesion_fraction,
        sliding_fraction,
        available_friction_budget,
        consumed_friction_budget,
        effective_longitudinal_stiffness,
        effective_cornering_stiffness,
        cells,
        dissipated_power,
    })
}
