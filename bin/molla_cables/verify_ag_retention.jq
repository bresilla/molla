$baseline[0] as $before | $current[0] as $after |
($ARGS.named.target // 0.20) as $target |
($ARGS.named.tolerance // 0.03) as $tolerance |
(1 - $after.shed / $before.shed) as $reduction |
{
  valid_target: ($target>0 and $target<1 and $tolerance>=0 and $tolerance<0.1),
  matching_duration: (($after.simulation_seconds - $before.simulation_seconds)|fabs)<0.01,
  matching_scene: (all(["trees","leaves","row_length_m","physics_hz","coupling","motors","ambient_wind","downwash_level","drone_mass_kg","subgrid_curl_strength","exterior_wake_decay_length_m","wood_flexibility"][]; . as $key | $after[$key]==$before[$key])),
  physical_strength_increased: ($after.leaf_strength_scale > $before.leaf_strength_scale),
  requested_reduction: (($reduction - $target|fabs) <= $tolerance),
  leaf_count_conserved: ($after.shed + $after.attached == $after.leaves),
  complete_route: ($after.spray.phase=="Complete" and ($after.spray.dose_seconds_per_tree|length)==16 and all($after.spray.dose_seconds_per_tree[]; .>=1.999 and .<=2.02)),
  intact_tree: ($after.root_error_m==0 and $after.wood_failures==0)
} as $checks |
{checks:$checks,passed:all($checks[];.),baseline_shed:$before.shed,shed:$after.shed,
 attached:$after.attached,leaf_strength_scale:$after.leaf_strength_scale,reduction_percent:100*$reduction,
 target_percent:100*$target,tolerance_percentage_points:100*$tolerance} |
if .passed then . else error("AG leaf-retention verification failed: \(.)") end
