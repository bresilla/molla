def transport:
  [.leaves[] | select(.attached == false)] as $shed |
  {
    total: ($shed | length),
    upwind_band: ([$shed[] | select(.position[0] > -5.1 and .position[0] < -4.7)] | length),
    back_band: ([$shed[] | select(.position[2] > -12.3 and .position[2] < -11.9)] | length),
    slow: ([$shed[] | select((.velocity | map(. * .) | add | sqrt) < 0.08)] | length),
    minimum_x_m: ($shed | map(.position[0]) | min),
    minimum_z_m: ($shed | map(.position[2]) | min)
  };
def abs: if . < 0 then -. else . end;
$before[0] as $b | $after[0] as $a |
($first[0] | transport) as $t0 | ($last[0] | transport) as $t1 |
(1 - $a.shed / $b.shed) as $reduction |
{
  shedding_reduction: $reduction,
  baseline: $t0,
  current: $t1,
  verified: (
    ($a.simulation_seconds - $b.simulation_seconds | abs) < 1e-6 and
    $a.leaves == $b.leaves and $a.shed == $t1.total and $b.shed == $t0.total and
    $reduction >= 0.15 and $reduction <= 0.25 and
    $a.airflow_visible == false and $a.downwash_level == $b.downwash_level and
    $a.wood_flexibility == $b.wood_flexibility and
    ($a.peak_trunk_displacement_m - $b.peak_trunk_displacement_m | abs) < 0.01 and
    ($a.peak_limb_displacement_m - $b.peak_limb_displacement_m | abs) < 0.02 and
    $a.root_error_m == 0 and $a.wood_failures == 0 and
    $a.air_divergence_rms < 0.01 and $a.ground_air_flux_max == 0 and
    $t1.upwind_band / $t1.total < 0.25 * $t0.upwind_band / $t0.total and
    $t1.back_band / $t1.total < 0.25 * $t0.back_band / $t0.total and
    $t1.minimum_x_m < -6 and $t1.minimum_z_m < -13 and
    all($last[0].leaves[]; .position[1] >= 0 and all(.position[], .velocity[]; isfinite))
  )
} | if .verified then . else error("leaf transport comparison failed: \(.)") end
