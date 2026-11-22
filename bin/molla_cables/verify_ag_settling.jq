$baseline[0] as $before | $current[0] as $after |
$before.spray as $old | $after.spray as $new |
{
  full_duration: (($after.simulation_seconds - 140)|fabs)<0.01 and (($after.simulation_seconds - $before.simulation_seconds)|fabs)<0.01,
  unchanged_air_and_leaves: (all(["trees","leaves","shed","attached","leaf_strength_scale","physics_hz","coupling","motors","ambient_wind","downwash_level","drone_mass_kg","subgrid_curl_strength","wood_flexibility"][]; . as $key | $after[$key]==$before[$key])),
  unchanged_flight: ($after.flight_dynamics==$before.flight_dynamics),
  unchanged_nozzle_and_turbulence: (all(["pressure_ratio","flow_l_s","downward_exit_gain","cone_full_angle_deg","parcel_rate_per_s","dispersion_intensity","dispersion_correlation_s","emitted_litres"][]; . as $key | $new[$key]==$old[$key])),
  physical_settling: ($new.diameter_um_range==[160,240] and $new.settling.gravity_m_s2==9.81),
  complete: ($new.phase=="Complete"),
  mostly_downward: ($new.settling.upward_volume_time_fraction<0.25),
  limited_rise: ($new.settling.max_rise_above_nozzle_m<0.5),
  reduced_residual_mist: ($new.airborne_litres<$old.airborne_litres*0.5),
  reduced_drift: ($new.max_horizontal_drift_m<$old.max_horizontal_drift_m*0.5),
  valid_measurements: (all([$new.settling.airborne_volume_time_l_s,$new.settling.upward_volume_time_l_s,$new.settling.max_rise_above_nozzle_m,$new.settling.upward_volume_time_fraction][];type=="number" and isfinite and .>=0) and $new.settling.airborne_volume_time_l_s>0 and $new.settling.upward_volume_time_l_s<=$new.settling.airborne_volume_time_l_s and (($new.settling.upward_volume_time_fraction-$new.settling.upward_volume_time_l_s/$new.settling.airborne_volume_time_l_s)|fabs)<1e-10)
} as $checks |
{checks:$checks,passed:all($checks[];.),
 baseline_maximum_drift_m:$old.max_horizontal_drift_m,maximum_drift_m:$new.max_horizontal_drift_m,
 baseline_airborne_litres:$old.airborne_litres,airborne_litres:$new.airborne_litres,
 settling:$new.settling} |
if .passed then . else error("AG settling verification failed: \(.)") end
