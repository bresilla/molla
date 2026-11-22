def norm: map(. * .) | add | sqrt;
$report[0] as $r | $state[0] as $s |
{
  seconds: $r.simulation_seconds,
  shed: $r.shed,
  retained: $r.attached,
  unique_relaunched_leaves: $r.unique_relaunched_leaves,
  relaunches: $r.leaf_relaunches,
  rising_steps: $r.airborne_rising_steps,
  rolling_steps: $r.ground_rolling_steps,
  original_full_row_gate: $r.verified_full_row,
  original_retention_target_met: ($r.attached > $r.leaves * 0.75),
  verified_motion: (
    $r.simulation_seconds >= 77.999999 and $r.simulation_seconds < 78.000001 and
    $r.airflow_visible == false and $r.downwash_level == 60 and
    $r.wood_flexibility == 4 and $r.subgrid_curl_strength > 0 and
    $r.leaf_strength_scale >= 0.57 and $r.root_error_m == 0 and $r.wood_failures == 0 and
    $r.unique_relaunched_leaves >= 25 and $r.leaf_relaunches >= $r.unique_relaunched_leaves and
    $r.airborne_rising_steps > 100 and $r.ground_rolling_steps > 1000 and
    $r.air_divergence_rms < 0.01 and $r.ground_air_flux_max == 0 and
    all($r.visited_canopy_sides[]; . == true) and
    all($r.tree_overflight_horizontal_gaps_m[]; . < 0.3) and
    ($r.drone_velocity | norm) < 0.1 and
    $r.furthest_drone_z_m > 8.3 and
    ([range(0;3) as $i | $r.drone_position[$i] - $r.flight_trace_txyz[0][$i+1]] | norm) < 0.1 and
    ($s.leaves | length) == $r.leaves and
    ([$s.leaves[] | select(.attached == false)] | length) == $r.shed and
    all($s.leaves[]; .position[1] >= 0 and all(.position[],.velocity[]; isfinite)) and
    all($s.wood[]; all(.center[],.velocity[]; isfinite))
  )
} | if .verified_motion then . else error("full-flight leaf motion verification failed: \(.)") end
